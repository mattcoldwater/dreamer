import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
from tqdm import tqdm
import time

from replay import initialize_replay_buffer, samples_to_buffer
from models.rnns import get_feat, get_dist, RSSMState
from models.utils import video_summary, get_parameters, FreezeParameters

# torch.autograd.set_detect_anomaly(True)  # used for debugging gradients

loss_info_fields = ['world_loss', 'actor_loss', 'value_loss', 'prior_entropy', 'post_entropy', 'divergence',
                    'reward_loss', 'image_loss', 'pcont_loss']
LossInfo = namedarraytuple('LossInfo', loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ['loss', 'grad_norm_world', 'grad_norm_actor', 'grad_norm_value'] + loss_info_fields)

def get_debuge_time(t, tn):
    t += time.time()
    print(tn, t)
    return -time.time(), tn+1

class Dreamer(RlAlgorithm):

    def __init__(
            self,  # Hyper-parameters
            batch_size=50,
            batch_length=50,
            train_every=1000,
            C=100,
            pretrain=100,
            world_lr=6e-4,
            value_lr=8e-5,
            actor_lr=8e-5,
            grad_clip=100.0,
            dataset_balance=False,
            discount=0.99,
            discount_lambda=0.95,
            horizon=15,
            action_dist='tanh_normal',
            action_init_std=5.0,
            expl='additive_gaussian',
            expl_amount=0.3,
            expl_decay=0.0,
            expl_min=0.0,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            replay_size=int(5e6),
            replay_ratio=8,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only. (not implemented)
            free_nats=3,
            kl_scale=1,
            type=torch.float,
            prefill=5000,
            log_video=True,
            video_every=int(1e1),
            video_summary_t=25,
            video_summary_b=4,
            use_pcont=False,
            discount_scale=10.0,
    ):
        super().__init__()
        if optim_kwargs is None:
            optim_kwargs = {}
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

        self.optimizer = None
        self.type = type

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        self.agent = agent
        self.n_itr = n_itr # n_steps
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1):
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec, async_=True)

    def optim_initialize(self, rank=0):
        self.rank = rank
        model = self.agent.model
        self.world_modules = [model.observation_encoder,
                              model.observation_decoder,
                              model.reward_model,
                              model.representation,
                              model.transition]
        if self.use_pcont:
            self.world_modules += [model.pcont]
        self.actor_modules = [model.action_decoder]
        self.value_modules = [model.value_model]
        self.world_optimizer = torch.optim.Adam(get_parameters(self.world_modules), lr=self.world_lr,
                                                **self.optim_kwargs)
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor_modules), lr=self.actor_lr,
                                                **self.optim_kwargs)
        self.value_optimizer = torch.optim.Adam(get_parameters(self.value_modules), lr=self.value_lr,
                                                **self.optim_kwargs)

        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        # must define these fields to for logging purposes. Used by runner.
        self.opt_info_fields = OptInfo._fields

    def optim_state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
                multiple optimizers."""
        return dict(
            world_optimizer_dict=self.world_optimizer.state_dict(),
            actor_optimizer_dict=self.actor_optimizer.state_dict(),
            value_optimizer_dict=self.value_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.world_optimizer.load_state_dict(state_dict['world_optimizer_dict'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer_dict'])
        self.value_optimizer.load_state_dict(state_dict['value_optimizer_dict'])

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr
        if samples is not None:
            # Note: discount not saved here
            self.replay_buffer.append_samples(samples_to_buffer(samples))

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        ####################### Fill the Buffer ############################
        if itr < self.prefill:
            return opt_info
        ####################### for time step t = 1..T do ############################
        # add experience to dataset D
        if itr % self.train_every != 0:
            return opt_info
        ####################### for update step c=1..C do ############################
        debug_time, debug_n = -time.time(), 0
        # for c in tqdm(range(self.C), desc='update step'):
        for c in range(self.C):
            # Draw B data sequences from D
            samples_from_replay = self.replay_buffer.sample_batch(self._batch_size, self.batch_length)
            buffed_samples = buffer_to(samples_from_replay, self.agent.device)

            world_loss, actor_loss, value_loss, loss_info = self.loss(buffed_samples, itr, c) # 9.488(2.54) seconds; 6.4-->5.5-->2.85(0.524) (light)
            # debug_time, debug_n = get_debuge_time(debug_time, debug_n)

            # Update θ(representation model), φ(action model), ψ(value model)
            self.world_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            world_loss.backward() # 11.668(2.7) seconds; 11.5-->9.2-->4.0(0.265) (gpu)
            # debug_time, debug_n = get_debuge_time(debug_time, debug_n)
            actor_loss.backward() # 1.14(0.122) seconds
            # debug_time, debug_n = get_debuge_time(debug_time, debug_n)
            value_loss.backward() # 0.171(0.0012) seconds
            # debug_time, debug_n = get_debuge_time(debug_time, debug_n)

            grad_norm_world = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_modules), self.grad_clip)
            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_modules), self.grad_clip)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_modules), self.grad_clip)

            self.world_optimizer.step()
            self.actor_optimizer.step()
            self.value_optimizer.step()

            ########################### rest is logging ##################################
            with torch.no_grad():
                loss = world_loss + actor_loss + value_loss
            opt_info.loss.append(loss.item())
            if isinstance(grad_norm_world, torch.Tensor):
                opt_info.grad_norm_world.append(grad_norm_world.item())
                opt_info.grad_norm_actor.append(grad_norm_actor.item())
                opt_info.grad_norm_value.append(grad_norm_value.item())
            else:
                opt_info.grad_norm_world.append(grad_norm_world)
                opt_info.grad_norm_actor.append(grad_norm_actor)
                opt_info.grad_norm_value.append(grad_norm_value)
            for field in loss_info_fields:
                if hasattr(opt_info, field):
                    getattr(opt_info, field).append(getattr(loss_info, field).item())

            self.agent._itr += 1
            
        return opt_info

    def loss(self, samples: SamplesFromReplay, sample_itr: int, opt_itr: int):
        """
        Compute the loss for a batch of data.  This includes computing the model and reward losses on the given data,
        as well as using the dynamics model to generate additional rollouts, which are used for the actor and value
        components of the loss.
        :param samples: samples from replay
        :param sample_itr: sample iteration, itr in range(5e6), aka-->step
        :param opt_itr: optimization iteration, for i in range(self.C), desc='update step'
        :return: FloatTensor containing the loss
        """
        model = self.agent.model

        observation = samples.all_observation[:-1]  # [t, t+batch_length+1] -> [t, t+batch_length]
        action = samples.all_action[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = samples.all_reward[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = reward.unsqueeze(2)
        done = samples.done
        done = done.unsqueeze(2)

        # Extract tensors from the Samples object
        # They all have the batch_t dimension first, but we'll put the batch_b dimension first.
        # Also, we convert all tensors to floats so they can be fed into our models.

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        # squeeze batch sizes to single batch dimension for imagination roll-out
        batch_size = batch_t * batch_b

        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(observation) # (50, 50, 1024) embed_size=1024 # 1.23 seconds

        prev_state = model.representation.initial_state(batch_b, device=action.device, dtype=action.dtype) # mean: (50, 30) stochastic_size=30 # 0.00014 s
        # Rollout model by taking the same series of actions as the real model # 1.75 s
        prior, post = model.rollout.rollout_representation(batch_t, embed, action, prev_state) # qθ(s t|s t−1, a t−1), pθ(s t|s t−1, a t−1, o t) RSSM mean,std..(50, 50, 30)
        # Flatten our data (so first dimension is batch_t * batch_b = batch_size)
        # since we're going to do a new rollout starting from each state visited in each batch.

        # Compute losses for each component of the model

        # Model Loss ################# Reconstruction loss ######################
        feat = get_feat(post) # (50, 50, 230) [post.stochahtic, post.deterministic] # 0.0002 s
        image_pred = model.observation_decoder(feat) # Normal(50, 50, 1, 64, 64) # 4.46 seconds
        reward_pred = model.reward_model(feat) # Normal(50, 50, 1) # 0.06 s
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(observation))
        pcont_loss = torch.tensor(0.)  # placeholder if use_pcont = False

        # It's just a different name for discount, the probability of the episode continuing. 
        # For episodic tasks, the agent needs to learn this to take it into account during imagination training.
        # We predict the discount factor from the latent state with a binary classiﬁer that is trained towards the soft labels of 0 and γ.
        if self.use_pcont:
            pcont_pred = model.pcont(feat) # discount predict, Bernoulli(50, 50, 1)
            pcont_target = self.discount * (1 - done.float())
            pcont_loss = -torch.mean(pcont_pred.log_prob(pcont_target))
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))
        world_loss = self.kl_scale * div + reward_loss + image_loss
        if self.use_pcont:
            world_loss += self.discount_scale * pcont_loss

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        #######################
        # argu_observation --> embed' --> contrastive loss(embed, embed')=auxilirary loss=moco_loss
        # loss = loss + (moco_loss * self.coeff)
        # self.online_net.zero_grad()
        # curl_loss = (weights * loss).mean()
        # curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        #######################

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # Don't let gradients pass through to prevent overwriting gradients.
        # Actor Loss ################# V_lambda ######################

        # remove gradients from previously calculated tensors
        with torch.no_grad():
            if self.use_pcont:
                # "Last step could be terminal." Done in TF2 code, but unclear why
                flat_post = buffer_method(post[:-1, :], 'reshape', (batch_t - 1) * (batch_b), -1) # RSSM mean..(2450, 30)
            else:
                flat_post = buffer_method(post, 'reshape', batch_size, -1)
        # Rollout the policy for self.horizon steps. Variable names with imag_ indicate this data is imagined not real.
        # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
        with FreezeParameters(self.world_modules): # 0.84 seconds
            imag_dist, _ = model.rollout.rollout_policy(self.horizon, model.policy, flat_post) # RSSM mean..(10, 2450, 30)

        # Use state features (deterministic and stochastic) to predict the image and reward # 0.009 s
        imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size] (10, 2450, 230)
        # Assumes these are normal distributions. In the TF code it's be mode, but for a normal distribution mean = mode
        # If we want to use other distributions we'll have to fix this.
        # We calculate the target here so no grad necessary

        # freeze model parameters as only action model gradients needed # 0.28 seconds
        with FreezeParameters(self.world_modules + self.value_modules):
            imag_reward = model.reward_model(imag_feat).mean
            value = model.value_model(imag_feat).mean
        # Compute the exponential discounted sum of rewards
        if self.use_pcont:
            with FreezeParameters([model.pcont]): # 0.17 seconds
                discount_arr = model.pcont(imag_feat).mean
        else:
            discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = self.compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1],
                                      bootstrap=value[-1], lambda_=self.discount_lambda) # (9, 2450, 1)
        # Make the top row 1 so the cumulative product starts with discount^0
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]]) # (10, 2450, 1)
        discount = torch.cumprod(discount_arr[:-1], 0) # (9, 2450, 1)
        actor_loss = -torch.mean(discount * returns)

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # Don't let gradients pass through to prevent overwriting gradients.
        # Value Loss  ################# |V_phi-V_lambda| ######################

        # remove gradients from previously calculated tensors
        with torch.no_grad():
            value_feat = imag_feat[:-1].detach()
            value_discount = discount.detach()
            value_target = returns.detach()
        value_pred = model.value_model(value_feat) # RSSM mean..(9, 2450, 1） 
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # loss info
        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())
            loss_info = LossInfo(world_loss, actor_loss, value_loss, prior_ent, post_ent, div, reward_loss, image_loss,
                                 pcont_loss)

            if self.log_video:
                if opt_itr == self.C - 1 and sample_itr % self.video_every == 0: # at update_dreamer_itr=C-1, step%10==0
                    self.write_videos(observation, action, image_pred, post, step=sample_itr, n=self.video_summary_b,
                                      t=self.video_summary_t) # (50, 50, 1, 64, 64) (50, 50, 6) 
                    print("Write videos|step =",sample_itr)

        return world_loss, actor_loss, value_loss, loss_info

    def write_videos(self, observation, action, image_pred, post, step=None, n=4, t=25):
        """
        observation shape T,N,C,H,W
        generates n rollouts with the model.
        For t time steps, observations are used to generate state representations.
        Then for time steps t+1:T, uses the state transition model.
        Outputs 3 different frames to video: ground truth, reconstruction, error
        """
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3) # 2, T=50, 50, (1,64,64)
        model = self.agent.model
        ground_truth = observation[:, :n] + 0.5 # (50, 4, 1, 64, 64), 0:T time steps
        reconstruction = image_pred.mean[:t, :n] # (25, 4, 1, 64, 64),  0:t time steps

        prev_state = post[t - 1, :n]
        prior = model.rollout.rollout_transition(batch_t - t, action[t:, :n], prev_state)
        imagined = model.observation_decoder(get_feat(prior)).mean # (25, 4, 1, 64, 64), t:T time steps
        model = torch.cat((reconstruction, imagined), dim=0) + 0.5 # (50, 4, 1, 64, 64), 0:T time steps
        error = (model - ground_truth + 1) / 2 # (50, 4, 1, 64, 64)
        # concatenate vertically on height dimension
        openl = torch.cat((ground_truth, model, error), dim=3) # (50, 4, 1, 192, 64)
        openl = openl.transpose(1, 0)  # N,T,C,H,W  (4, 50, 1, 192, 64)
        video_summary('videos/world_model_error', torch.clamp(openl, 0., 1.), step)

    def compute_return(self,
                       reward: torch.Tensor,
                       value: torch.Tensor,
                       discount: torch.Tensor,
                       bootstrap: torch.Tensor,
                       lambda_: float):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns
