import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class SymbolicBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            hard_epoch=2500,
            warmup_epoch=0,
            target_temp=0.2,
            spls=0.1, 
            target_ratio=0.1,
            constrain_scale=1,
            l0_scale=0.01,
            bl0_scale=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.target_ratio = target_ratio
        self.hard_epoch = hard_epoch
        self.warmup_epoch = warmup_epoch
        self.target_temp = target_temp
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        #self.spls*self.sparse_loss+self.constrain_loss*self.constrain_scale+self.regu_loss+self.l0_scale*self.l0_loss+self.bl0_scale*self.bl0_loss
        self.trainer.policy.symbolic.spls= spls
        self.trainer.policy.symbolic.constrain_scale  = constrain_scale
        self.trainer.policy.symbolic.l0_scale = l0_scale
        self.trainer.policy.symbolic.bl0_scale = bl0_scale

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            self.trainer.policy.symbolic.update_const()
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.trainer.policy.symbolic.update_const()
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')
        self.trainer.policy.symbolic.temp=1/((1-self.target_temp)*(1 - min(self.epoch,self.hard_epoch)/self.hard_epoch) + self.target_temp)
        clip_it= max(min(self.epoch,self.hard_epoch),self.warmup_epoch)
        self.trainer.policy.symbolic.target_ratio = self.target_ratio+(1-self.target_ratio)*(1-((clip_it-self.warmup_epoch)/(self.hard_epoch-self.warmup_epoch))**2)
        for _ in range(self.num_train_loops_per_epoch):
            self.trainer.policy.symbolic.update_const()
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)

            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)

            self.training_mode(False)
