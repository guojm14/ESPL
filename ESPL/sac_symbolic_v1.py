from gym.envs.mujoco import HopperEnv,Walker2dEnv,InvertedPendulumEnv
from gym.envs.box2d import LunarLanderContinuous
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianSymbolicPolicy_V1, MakeDeterministic
from rlkit.torch.sac.sac_symbolic import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import SymbolicTorchBatchRLAlgorithm
import torch
import numpy as np

def experiment(variant):
    expl_env = NormalizedBoxEnv(env_dict[variant['env_name']]())
    eval_env = NormalizedBoxEnv(env_dict[variant['env_name']]())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianSymbolicPolicy_V1(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = SymbolicTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


import argparse
env_dict={'lunar_lander':LunarLanderContinuous}
if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='PEARL')
    parser.add_argument('--env', type=str, default="cheetah")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hard_epoch', type=int, default=900)
    parser.add_argument('--spls', type=float, default=0.1)
    parser.add_argument('--target_ratio', type=float, default=0.001)

    args = parser.parse_args()

    variant = dict(
        algorithm="SACsymbolicv1",
        version="normal",
        layer_size=256,
        env_name= args.env,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            hard_epoch=args.hard_epoch,
            warmup_epoch=0,
            target_temp=0.2,
            spls=args.spls, 
            target_ratio=args.target_ratio,
            constrain_scale=1,
            l0_scale=0,
            bl0_scale=0,            
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=5,
            use_automatic_entropy_tuning=True,
            sample_num=2,
        ),
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed )
    setup_logger('sacv1_'+args.env+'he'+str(args.hard_epoch)+'sp'+str(args.spls)+'tr'+str(args.target_ratio), variant=variant,seed=args.seed)
    ptu.set_gpu_mode(True)  
    experiment(variant)

