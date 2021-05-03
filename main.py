import datetime
import os
import argparse
import torch

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
# Runner - Connects the sampler, agent, and algorithm; manages the training loop and logging of diagnostics
from rlpyt.samplers.serial.sampler import SerialSampler
# Sampler - Manages agent / environment interaction to collect training data, can initialize parallel workers
from rlpyt.utils.logging.context import logger_context

from dreamer_agent import AtariDreamerAgent
from dreamer_algo import Dreamer
from envs.atari import AtariEnv, AtariTrajInfo
from envs.wrapper import make_wapper
from envs.one_hot import OneHotAction
from envs.time_limit import TimeLimit


def build_and_train(log_dir, game="pong", run_ID=0, cuda_idx=None, eval=False, save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2
    env_kwargs = dict(
        name=game,
        action_repeat=action_repeat,
        size=(64, 64),
        grayscale=False,
        life_done=True,
        sticky_actions=True,
    )
    factory_method = make_wapper(
        AtariEnv,
        [OneHotAction, TimeLimit],
        [dict(), dict(duration=1000 / action_repeat)])
    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore # TrajectoryInfo - Diagnostics logged on a per-trajectory basis.
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = Dreamer(horizon=10, kl_scale=0.1, use_pcont=True, initial_optim_state_dict=optimizer_state_dict)
    # algo = Dreamer(horizon=10, kl_scale=0.1, use_pcont=True, initial_optim_state_dict=optimizer_state_dict, replay_size=int(6e5)) # haoyu replay_size=int(5e6)
    agent = AtariDreamerAgent(train_noise=0.4, eval_noise=0, expl_type="epsilon_greedy",
                              expl_min=0.1, expl_decay=2000 / 0.3, initial_model_state_dict=agent_state_dict,
                              model_kwargs=dict(use_pcont=True))
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo, # Uses gathered samples to train the agent (e.g. defines a loss function and performs gradient descent).
        agent=agent, # Chooses control action to the environment in sampler; trained by the algorithm. Interface to model.
        sampler=sampler,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    # python main.py --log-dir /content/debug/
    # python main.py --log-dir /content/pong_0/ --game pong --cuda-idx 0
    # python main.py --log-dir /content/chopper_command_0/ --game chopper_command --cuda-idx 0
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong') # games = ["pong", "chopper_command"]
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl

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
        load_model_path=args.load_model_path
    )
