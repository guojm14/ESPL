import numpy as np
from rand_param_envs.cartpole_rand_params import CartPoleEnv

from . import register_env


@register_env('cartpole-rand-params-ood')
class CartPoleWrappedEnv(CartPoleEnv):
    def __init__(self, n_tasks=2, n_train_tasks=40,n_eval_tasks=10,randomize_tasks=True):
        super(CartPoleWrappedEnv, self).__init__()
        self.tasks = self.sample_ood_tasks(n_train_tasks,n_eval_tasks)
        print(self.tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
    
