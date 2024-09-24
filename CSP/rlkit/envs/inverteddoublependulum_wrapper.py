import numpy as np
from rand_param_envs.inverteddoublependulum_rand import InvertedDoublePendulumEnv

from . import register_env


@register_env('inverteddoublependulum-rand-params')
class InvertedDoublePendulumWrappedEnv(InvertedDoublePendulumEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True):
        super(InvertedDoublePendulumWrappedEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        print(self.tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
    
