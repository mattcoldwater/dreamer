import datetime
import os
import argparse
import torch

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
# Runner - Connects the sampler, agent, and algorithm; manages the training loop and logging of diagnostics
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
# Sampler - Manages agent / environment interaction to collect training data, can initialize parallel workers
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.seed import set_seed

from dreamer_agent import AtariDreamerAgent
from dreamer_algo import Dreamer
from envs.atari import AtariEnv, AtariTrajInfo
from envs.wrapper import make_wapper
from envs.one_hot import OneHotAction
from envs.time_limit import TimeLimit


def build_and_train(log_dir, game="pong", run_ID=0, cuda_idx=None, eval=False, save_model='last', load_model_path=None, n_parallel=2, CumSteps=0):
    device = 'cpu' if cuda_idx is None else 'cuda'
    params = torch.load(load_model_path, map_location=torch.device(device)) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    ##--- wu ---##
    log_interval_steps = 5e4
    prefill = 5e4
    train_every = 16
    batch_B = 16
    n_steps = 1e4 if eval else 5e6
    itr_start = max(0, CumSteps - prefill) // train_every
    ##--- wu ---##

    action_repeat = 4 # 2
    env_kwargs = dict(
        name=game,
        action_repeat=action_repeat,
        size=(64, 64),
        grayscale=True, # False
        life_done=True,
        sticky_actions=True,
    )
    factory_method = make_wapper(
        AtariEnv,
        [OneHotAction, TimeLimit],
        [dict(), dict(duration=1000000 / action_repeat)]) # 1000
    
    sampler = GpuSampler(
        EnvCls=factory_method,
        TrajInfoCls=AtariTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        batch_T=1,
        batch_B=batch_B,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e5),
        eval_max_trajectories=5,
    )

    algo = Dreamer(initial_optim_state_dict=optimizer_state_dict,
            horizon=10,
            use_pcont=True,
            replay_size=int(2e6), # int(5e6)
            kl_scale=0.1, 
            batch_size=50,
            batch_length=50,
            C=1, # 100,
            train_every=train_every // batch_B, # 1000
            pretrain=100,
            world_lr=2e-4, # 6e-4,
            value_lr=1e-4, # 8e-5,
            actor_lr=4e-5, # 8e-5,
            discount=0.999, # 0.99,
            expl_amount=0.0, # 0.3,
            prefill=prefill // batch_B, # 5000
            discount_scale=5., # 10.
            video_every=int(2e4 // 16 * 16 // batch_B), # int(10)
        )
    
    if eval:
        # for eval - all versions
        agent = AtariDreamerAgent(train_noise=0.0, eval_noise=0, expl_type="epsilon_greedy", itr_start=itr_start, the_expl_mode='eval',
                                expl_min=0.0, expl_decay=11000, initial_model_state_dict=agent_state_dict,
                                model_kwargs=dict(use_pcont=True))
    else:
        # for train - all versions
        # agent = AtariDreamerAgent(train_noise=0.4, eval_noise=0, expl_type="epsilon_greedy", itr_start=itr_start, the_expl_mode='train',
        #                           expl_min=0.1, expl_decay=11000, initial_model_state_dict=agent_state_dict,
        #                           model_kwargs=dict(use_pcont=True))

        # for train - dreamer_V2
        agent = AtariDreamerAgent(train_noise=0.0, eval_noise=0, expl_type="epsilon_greedy", itr_start=itr_start, the_expl_mode='train',
                                expl_min=0.0, expl_decay=11000, initial_model_state_dict=agent_state_dict,
                                model_kwargs=dict(use_pcont=True))

    my_seed = 0 # reproductivity
    set_seed(my_seed)
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo, # Uses gathered samples to train the agent (e.g. defines a loss function and performs gradient descent).
        agent=agent, # Chooses control action to the environment in sampler; trained by the algorithm. Interface to model.
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=log_interval_steps,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel))),
        seed=my_seed,
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    # python main.py --log-dir /content/debug/ --game pong --load-model-path '/content/pong_0/run_2/itr_149999.pkl' --eval
    # python main.py --log-dir /content/pong_0/ --game pong --cuda-idx 0 # --load-model-path '/gdrive/MyDrive/CSE525/Project/saved/itr_68999.pkl'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong') # games = ["pong", "chopper_command"]
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument('--CumSteps', help='CumSteps', type=int, default=0)

    default_log_dir = os.path.join(
        '/content/',
        # os.path.dirname(__file__),
        'data',
        'local',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        print(f'run {i} already exists. ')
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    build_and_train(
        log_dir,
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        eval=args.eval,
        save_model=args.save_model,
        load_model_path=args.load_model_path,
        n_parallel=args.n_parallel,
        CumSteps=args.CumSteps
    )

"""
buffer fill 5e2
    under small net:
    C=50, every=500 128s; 
    C=1, every=16 80s-->padding-->35s.
    under big net:
    C=1, every=16 97s.

    fill 480
    under small net:
    -- 2 cpu -gpusampler B=2 18*2 seconds
    -- serial mode 34 seconds
buffer fill 5e3 in gpu serial mode
    under small net:
    05:56 seconds
    0.00859s/step - env
    0.2003s/step - Dreamer
buffer fill 5e4 in cpu core 4
     00:04:40 para=1 B=1 CumSteps 50000 NewCompletedTrajs 28 StepsPerSecond       177.964
     00:02:15 para=3 B=3 CumSteps 49998 NewCompletedTrajs 27 StepsPerSecond       365.73
     para=4 B=4 StepsPerSecond        370.25 
     para=4 B=16 StepsPerSecond       697.326
     para=4 B=10 T=10 StepsPerSecond  864.995
     para=4 B=10 T=13 StepsPerSecond       876.363
     para=4 B=10 T=16 StepsPerSecond       978.95
     para=4 B=10 T=17 StepsPerSecond       947.888
     para=4 B=10 T=25 StepsPerSecond       911.478

https://github.com/danijar/dreamerv2
  # General
  task: 'atari_pong'
  steps: 2e8
  eval_every: 1e5
  log_every: 1e4
  prefill: 50000
  dataset_size: 2e6
  pretrain: 0

  # Environment
  time_limit: 108000  # 30 minutes of game play.
  grayscale: True
  action_repeat: 4
  eval_noise: 0.0
  train_every: 16
  train_steps: 1
  clip_rewards: 'tanh'

  # Model
  grad_heads: ['image', 'reward', 'discount']
  dyn_cell: 'gru_layer_norm'
  pred_discount: True
  cnn_depth: 48
  dyn_deter: 600
  dyn_hidden: 600
  dyn_stoch: 32
  dyn_discrete: 32
  reward_layers: 4
  discount_layers: 4
  value_layers: 4
  actor_layers: 4

  # Behavior
  actor_dist: 'onehot'
  actor_entropy: '1e-3'
  expl_amount: 0.0
  discount: 0.999
  imag_gradient: 'reinforce'
  imag_gradient_mix: '0'

  # Training
  discount_scale: 5.0
  reward_scale: 1
  weight_decay: 1e-6
  model_lr: 2e-4
  kl_scale: 0.1
  kl_free: 0.0
  actor_lr: 4e-5
  value_lr: 1e-4
  oversample_ends: True
"""