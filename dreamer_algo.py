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

from replay import initialize_replay_buffer, samples_to_buffer
from models.rnns import get_feat, get_dist, RSSMState
from models.utils import video_summary, get_parameters, FreezeParameters

loss_info_fields = ['world_loss', 'actor_loss', 'value_loss', 'prior_entropy', 'post_entropy', 'divergence',
                    'reward_loss', 'image_loss', 'pcont_loss']
LossInfo = namedarraytuple('LossInfo', loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ['loss', 'grad_norm_world', 'grad_norm_actor', 'grad_norm_value'] + loss_info_fields)

import time
def get_debuge_time(t, tn):
    t += time.time()
    print(tn, t)
    return -time.time(), tn+1

class Dreamer(RlAlgorithm):

    def __init__(
            self,  # Default Hyper-parameters
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
            updates_per_sync=1,
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

    ## init
    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        self.agent = agent
        self.n_itr = n_itr # n_steps
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec)
        self.optim_initialize(rank)
        assert self.use_pcont == True

    def optim_initialize(self, rank=0):
        self.rank = rank
        model = self.agent.model
        self.world_modules = [model.observation_encoder,
                              model.observation_decoder,
                              model.reward_model,
                              model.representation,
                              model.transition,
                              model.pcont]
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

    ## main loop
    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr
        ####################### for time step t = 1..T do ############################
        ####################### Environment interaction #######################
        if samples is not None: # add experience to dataset D
            self.replay_buffer.append_samples(samples_to_buffer(samples))
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.prefill: # prefill buffer
            return opt_info
        if itr % self.train_every != 0: # t = 1 .. T
            return opt_info
        ####################### for update step c=1..C do ############################
        # for c in tqdm(range(self.C), desc='update step'):
        for c in range(self.C):
            # Draw B data sequences from D
            samples_from_replay = self.replay_buffer.sample_batch(self._batch_size, self.batch_length)
            buffed_samples = buffer_to(samples_from_replay, self.agent.device)

            # Dynamics learning
            losses_list, post_states, batch_t, batch_b, prior_ent, post_ent = self.dynamics_learning(buffed_samples, itr, c)
            [world_loss, kl_diverge_loss, observation_loss, reward_loss, discount_loss] = losses_list

            # Behavior learning
            actor_loss, value_loss = self.behavior_learning(post_states, batch_t, batch_b)

            ################### update model paameters #######################
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
                loss_info = LossInfo(world_loss, actor_loss, value_loss, prior_ent, post_ent, 
                                kl_diverge_loss, reward_loss, observation_loss, discount_loss)
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

            self.agent._itr += 1 # one gradient step
            
        return opt_info

    ## Dynamic
    def dynamics_learning(self, samples: SamplesFromReplay, sample_itr: int, opt_itr: int):
        """
        :param samples: samples from replay
        :param sample_itr: sample iteration, itr in range(5e6), aka-->step
        :param opt_itr: optimization iteration, for i in range(self.C), desc='update step'
        """
        observation, action, reward, done = samples.all_observation[:-1], samples.all_action[1:], samples.all_reward[1:], samples.done
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        observation = observation.type(self.type) / 255.0 - 0.5
        reward, done = reward.unsqueeze(2), done.unsqueeze(2)
        discount = self.discount * (1 - done.float())

        ################# World model θ ####################
        w_encoder = self.agent.model.observation_encoder
        w_decoder = self.agent.model.observation_decoder
        w_reward = self.agent.model.reward_model
        w_represent = self.agent.model.representation
        w_transition_represent = self.agent.model.rollout
        w_discount = self.agent.model.pcont

        ########## World (model) Loss ################
        ## Equation (10) variational information bottleneck ##

        # 1. J_D = −KL[p(st | st−1; at−1; ot), q(st | st−1; at−1)]
        # first state
        prev_state = w_represent.initial_state(batch_b, device=action.device, dtype=action.dtype) # mean: (50, 30) stochastic_size=30
        # use representation p θ (s t | s t−1 ,a t−1 ,o t ) to reconstruct every step --> post
        # use traision q θ (s t | s t−1 ,a t−1 ) to reconstruct every step --> prior
        prior_states, post_states = w_transition_represent.rollout_representation(batch_t, w_encoder(observation), action, prev_state) # qθ(s t|s t−1, a t−1), pθ(s t|s t−1, a t−1, o t) RSSM mean,std..(50, 50, 30)
        # KL Divergence
        prior_states_distribution = get_dist(prior_states)
        post_states_distribution = get_dist(post_states)
        kl_diverge_loss = self.KL_Diverge_Loss(post_states_distribution, prior_states_distribution)

        # 2. J_O = ln : q θ (ot | st)
        sto_det_feature = get_feat(post_states) # (50, 50, 230) [post.stochahtic, post.deterministic] feature, see ./models folder
        pred_observation = w_decoder(sto_det_feature) # Normal(50, 50, 1, 64, 64)
        observation_loss = -torch.mean(pred_observation.log_prob(observation))

        # 3. J_R = ln : q θ (rt | st)
        pred_reward = w_reward(sto_det_feature) # Normal(50, 50, 1)
        reward_loss = -torch.mean(pred_reward.log_prob(reward))

        # 4. predict the discount factor from the latent state with a binary classifier 
        #   that is trained towards the soft labels of 0 and γ
        pred_discount = w_discount(sto_det_feature) # discount predict, Bernoulli(50, 50, 1)
        discount_loss = -torch.mean(pred_discount.log_prob(discount))

        # J_REC = JO + JR + β*JD + s*JDis
        world_loss = self.kl_scale * kl_diverge_loss + reward_loss + observation_loss + self.discount_scale * discount_loss

        ########################### rest is logging ##################################
        with torch.no_grad():
            prior_states_dist_ent = torch.mean(prior_states_distribution.entropy())
            post_states_dist_ent = torch.mean(post_states_distribution.entropy())

            if self.log_video:
                if opt_itr == self.C - 1 and sample_itr % self.video_every == 0: # at update_dreamer_itr=C-1, step%10==0
                    self.write_videos(observation, action, pred_observation, post_states, step=sample_itr, 
                                    n=self.video_summary_b, t=self.video_summary_t) # (50, 50, 1, 64, 64) (50, 50, 6) 
                    print("Write videos|step =",sample_itr)

        losses_list = [world_loss, kl_diverge_loss, observation_loss, reward_loss, discount_loss]
        return losses_list, post_states, batch_t, batch_b, prior_states_dist_ent, post_states_dist_ent

    def imagine_trajectories(self, _initial_states: RSSMState, batch_t: int, batch_b: int):
        ############# Imagine trajectories ##########
        ########### {sτ ; aτ } from each st ##########

        # no gradient for input (initial) states
        with torch.no_grad():
            initial_states = buffer_method(_initial_states[:-1, :], 'reshape', (batch_t - 1) * (batch_b), -1) # RSSM mean..(2450, 30)
        
        # imagine trajectories with a finite horizon H
        w_transition_represent = self.agent.model.rollout
        policy = self.agent.model.policy
        with FreezeParameters(self.world_modules):
            imagined_states, _ = w_transition_represent.rollout_policy(self.horizon, policy, initial_states) # RSSM mean..(10, 2450, 30)
        
        return imagined_states

    def get_V_lambda(self, feature: torch.Tensor):
        # models
        w_reward = self.agent.model.reward_model
        w_discount = self.agent.model.pcont
        critic = self.agent.model.value_model

        # 2. Predict rewards E: qθ(rτ | sτ ) and values v (sτ )
        with FreezeParameters(self.world_modules + self.value_modules):
            # Freeze world & critic
            rewards = w_reward(feature).mean
            discounts = w_discount(feature).mean
            values = critic(feature).mean

        # V_λ: exponentially-weighted avg of the estimates for different k to balance bias and variance
        V_lambda = self.compute_return(rewards[:-1], values[:-1], discounts[:-1],
                                      bootstrap=values[-1], lambda_=self.discount_lambda) # (9, 2450, 1)

        # discounts = cumprod(discounts)
        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:]]) # (10, 2450, 1)
        discounts = torch.cumprod(discounts[:-1], 0) # (9, 2450, 1)

        return V_lambda, discounts

    def KL_Diverge_Loss(self, post_states_distribution, prior_states_distribution):
        KL_Diverge = kl_divergence(post_states_distribution, prior_states_distribution)
        KL_Diverge = torch.mean(KL_Diverge)
        KL_Diverge = torch.max(KL_Diverge, KL_Diverge.new_full(KL_Diverge.size(), self.free_nats))
        return KL_Diverge

    ## Behavior
    def behavior_learning(self, _initial_states: RSSMState, batch_t: int, batch_b: int):
        # 1.imagine trajectories | horizon H
        imagined_states = self.imagine_trajectories(_initial_states, batch_t, batch_b)
        # state --> feature
        imagined_sto_det_f = get_feat(imagined_states)  # [horizon, batch_t * batch_b, feature_size] (10, 2450, 230)

        # 2. Predict rewards E: qθ(rτ | sτ ) and values v (sτ )
        # 3. Compute value estimates Vλ(sτ ) via Equation 6.
        V_lambda, discounts = self.get_V_lambda(imagined_sto_det_f)

        ################# Actor Loss ######################
        # 4. Update φ; Equation (7)  max: Vλ(sτ) 
        actor_loss = -torch.mean(discounts * V_lambda)

        ################# Value Loss ######################
        # Official Tensorflow version, ln : q φ (Vλ(sτ) | st);  φ-action_model
        # not sure why, in paper, Equation (8)  min: ||vφ(sτ) − Vλ(sτ)||2
        #     value_pred = self._value(imag_feat)[:-1]
        #     target = tf.stop_gradient(returns)
        #     value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
        critic = self.agent.model.value_model
        with torch.no_grad():
            imagined_sto_det_f_0 = imagined_sto_det_f[:-1].detach()
            discounts_0 = discounts.detach()
            V_lambda_0 = V_lambda.detach()
        V_phi = critic(imagined_sto_det_f_0) # RSSM mean..(9, 2450, 1)
        value_loss = - torch.mean(discounts_0 * V_phi.log_prob(V_lambda_0).unsqueeze(2)) 

        return actor_loss, value_loss

    ## utility
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