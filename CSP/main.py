
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, Model, EmbeddingMLP
from pearl_configs.default import default_config
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.symbolic_sac import PEARLSoftActorCriticSymbolic
from rlkit.torch.sac.eql import TanhGaussianSymbolic

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
import os
import os.path as osp
import pathlib
import numpy as np
import gym
import argparse
import json
import torch
import socket
from rlkit.core import logger
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def experiment(variant):
    

    # create multi-task environment
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    torch.manual_seed(variant['seed'])
    np.random.seed(variant['seed'])
    env.seed(variant['seed'])


    # sample tasks
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks



    file_name = "%s_%s_scale%d_seed%d" % (
        'csp', variant['env_name'], variant['algo_params']['reward_scale'], variant['seed'])
    latent_dim = variant['latent_size']
    task_dim = variant['latent_size']
    print(task_dim)
    context_encoder_input_dim = 2 * obs_dim + action_dim + \
        reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + \
        action_dim + reward_dim
    context_encoder_output_dim = latent_dim * \
        2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    network_model = FlattenMlp

    policy_model = TanhGaussianSymbolic


    agent_algo = PEARLAgent
    meta_algo = PEARLSoftActorCriticSymbolic
    context_encoder = None

    print("pearl context")
    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,

    ).to(device)

    qf1 = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=task_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,

    ).to(device)
    qf2 = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=task_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,

    ).to(device)
    qf1_old = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=task_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,

    ).to(device)
    qf2_old = network_model(
        hidden_sizes=[net_size, net_size, net_size],
        output_size=1,
        obs_dim=obs_dim,
        latent_dim=task_dim,
        action_dim=action_dim,
        input_size=latent_dim+obs_dim+action_dim,
        z_dim=latent_dim,

    ).to(device)
    policy = policy_model(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim,
        latent_dim=task_dim,
        action_dim=action_dim,
        input_size=obs_dim+latent_dim,

        std_hidden_dim=variant['std_hidden_dim'],
        context_hidden_dim=variant['context_hidden_dim'],
        arch_index=variant['arch_index'],
    ).to(device)
    agent = agent_algo(
        task_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    algorithm = meta_algo(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, qf1_old, qf2_old],
        latent_dim=task_dim,
        **variant['algo_params'],
        file_name=file_name,
        regulate_loss=variant['regulate_loss'],
        warmup_epoch=variant['warmup_epoch'],
        target_ratio=variant['target_ratio'],
        hard_epoch=variant['hard_epoch'],
        init_temp=variant['init_temp'],
        spls=variant['spls'],
        constrain_scale = variant['constrain_scale'],
        grad_norm = variant['grad_norm'],
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(
            os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(
            torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'],
                     variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))
    print(ptu.device)
    # create logging directory
    # TODO support Docker
    exp_id = '1'

    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                      base_log_dir=variant['util_params']['base_log_dir'], seed=variant['seed'])
    logger.save_extra_data(env.tasks, path='env_tasks')
    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def main(args):
    variant = default_config
    if args.config:
        with open(osp.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
        variant['seed'] = args.seed
        variant['regulate_loss'] = args.regulate_loss
        variant['std_hidden_dim'] = args.std_hidden_dim
        variant['context_hidden_dim'] = args.context_hidden_dim
        variant['warmup_epoch'] = args.warmup_epoch
        variant['target_ratio'] = args.target_ratio
        variant['init_ratio'] = args.init_ratio

        variant['hard_epoch'] = args.hard_epoch


        variant['arch_index'] = args.arch_index

        variant['init_temp'] = args.init_temp
        variant['constrain_scale'] = args.constrain_scale
        variant['spls'] = args.spls

        variant['grad_norm'] = args.grad_norm
    experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PEARL')
    parser.add_argument('--config', type=str,
                        default='configs/cheetah-vel-hard.json')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--target_ratio', type=float, default=0.01)
    parser.add_argument('--init_ratio', type=float, default=0.99)

    parser.add_argument('--arch_index', type=int, default=1)

    parser.add_argument('--spls', type=float, default=1)
    parser.add_argument('--hard_epoch', type=int, default=100)
    parser.add_argument('--constrain_scale', type=float, default=0.001)
    parser.add_argument('--std_hidden_dim', type=int, default=64)
    parser.add_argument('--context_hidden_dim', type=int, default=64)

    parser.add_argument("--regulate_loss", action="store_true", default=False)
    parser.add_argument('--grad_norm', type=float, default=0.5)
    parser.add_argument('--init_temp', type=float, default=0.03)
    args = parser.parse_args()
    main(args)
