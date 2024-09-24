from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import logger, eval_util




class PEARLSoftActorCriticSymbolic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,

            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            regulate_loss = True,
            soft_target_tau=1e-2,

            plotter=None,
            render_eval_paths=False,
            hard_epoch=100,
            alpha=0.2,
            target_ratio = 0.01,
            sample_num = 2,
            warmup_epoch =0,
            init_temp=0.03,
            sche=2,
            spls = 1,
            l0_scale = 0.01,
            bl0_scale = 0,
            sim_scale = 0,
            constrain_scale =1,
            grad_norm=0,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        self.grad_norm = grad_norm
        self.init_temp = init_temp
        self.l0_scale = l0_scale
        self.bl0_scale = bl0_scale
        self.sim_scale = sim_scale
        self.constrain_scale = constrain_scale
        self.sche = sche
        self.spls = spls
        self.target_ratio = target_ratio
        self.sample_num = sample_num
        self.warmup_epoch = warmup_epoch
        self.regulate_loss=regulate_loss
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()

        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self._alpha=alpha
        
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.hard_epoch = hard_epoch
        self.l0_scale = l0_scale
        
        

        self.qf1, self.qf2, self.qf1_old, self.qf2_old= nets[1:]
        self.copy_params(self.qf1, self.qf1_old)
        self.copy_params(self.qf2, self.qf2_old)

        lr = qf_lr
        p_lr = policy_lr
        c_lr = context_lr

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=p_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=lr,
        )

        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=c_lr,
        )

       
    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.qf1_old, self.qf2_old]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        #print('batch',batch['terminals'].mean())
        return [o, a, r, no, t]

    def copy_params(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def disable_gradients(self, net):
        for p in net.parameters():
            p.requires_grad = False
	
    def enable_gradients(self, net):
        for p in net.parameters():
            p.requires_grad = True

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            
            
            self.agent.policy.symbolic.temp=1/((1-self.init_temp)*(1 - min(it_,self.hard_epoch)/self.hard_epoch) + self.init_temp)
            #self.agent.policy.symbolic.temp=(1-self.init_temp)*min(it_,self.hard_epoch)/self.hard_epoch + self.init_temp
            clip_it= max(min(it_,self.hard_epoch),self.warmup_epoch)
            self.sparse_ratio = self.target_ratio+(1-self.target_ratio)*(1-((clip_it-self.warmup_epoch)/(self.hard_epoch-self.warmup_epoch))**self.sche)
            self.sim_scale_= self.sim_scale*(clip_it-self.warmup_epoch)/(self.hard_epoch-self.warmup_epoch)
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    #print(idx)
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.

            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices,it_)
                self._n_train_steps_total += 1
                gt.stamp('train_step')
            gt.stamp('train')
            logger.save_extra_data(self.agent.policy.symbolic.constw_mask, path='constwmask'+str(it_))
            logger.save_extra_data(self.agent.policy.symbolic.scores, path='scores'+str(it_))
            if it_ > self.hard_epoch:
                logger.save_extra_data(self.agent.policy.symbolic.constw, path='constw'+str(it_))
                logger.save_extra_data(self.agent.policy.symbolic.constb, path='constb'+str(it_))
                
            self.training_mode(False)

            # eval
            self._try_to_eval(it_)

            gt.stamp('eval')

            #plot rewrads
            np.save("./results/%s_train" % (self.file_name), self.evaluations['train'])
            np.save("./results/%s_test" % (self.file_name), self.evaluations['test'])
            #torch.save(self, "./pytorch_models/%s" % (self.file_name))
                
            self._end_epoch()
    ##### Training #####
    def _do_training(self, indices,it_):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]

            self._take_step(indices, context,it_)
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_old, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_old, self.soft_target_tau)


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            qf1_old=self.qf1_old.state_dict(),
            qf2_old=self.qf2_old.state_dict(),       
            policy=self.agent.policy.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot


    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns= self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))

        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)



        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        # avg_test_return_m1 = np.mean(test_final_returns_m1)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)

        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        # self.eval_statistics['AverageReturn_all_test_tasks_m1'] = avg_test_return_m1
        self.evaluations['test'].append(avg_test_return)
        self.evaluations['train'].append(avg_train_return)

        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()


    def _take_step(self, indices, context,it_):
       
        num_tasks = len(indices)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks

        self.agent.policy.symbolic.batch=self.batch_size
        #self.agent.policy.mode=1
        policy_outputs, task_z = self.agent(next_obs, context)
        next_actions, _, _, next_log_pi = policy_outputs[:4]
        # end.record()
        # torch.cuda.synchronize()
        # print('inference_time',start.elapsed_time(end))         
        self.agent.policy.symbolic.mode=1
        self.agent.policy.mode=1
        policy_outputs, _ = self.agent.get_policy(obs)
        self.agent.policy.symbolic.mode=0
        self.agent.policy.mode=0

        
        count_l0= self.agent.policy.symbolic.expect_w()
        constrain_loss= self.agent.policy.symbolic.constrain_loss()

        regu_loss = self.agent.policy.regu_loss
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]


        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        with torch.no_grad():
            rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            # scale rewards for Bellman update
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = terms.view(self.batch_size * num_tasks, -1)
            #print(terms_flat.mean(),rewards_flat.mean())
            min_target= torch.min(self.qf1_old(next_obs,next_actions,task_z),self.qf2_old(next_obs,next_actions,task_z)) - self._alpha*next_log_pi
            q_target = rewards_flat + (1. - terms_flat) * self.discount * min_target


        q1_pred = self.qf1(obs, actions.detach(), task_z)
        q2_pred = self.qf2(obs, actions.detach(), task_z)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self.context_optimizer.step()
        # end.record()
        # torch.cuda.synchronize()
        # print('critic_time',start.elapsed_time(end))
 
        self._update_target_network()

        # end.record()
        # torch.cuda.synchronize()
        # print('critic_time',start.elapsed_time(end))
        self.disable_gradients(self.qf1)
        self.disable_gradients(self.qf2)
        
        if self.sample_num>1:
            obs=obs.unsqueeze(0).expand(self.sample_num,-1,-1).reshape(self.sample_num*t*b,-1)
            task_z=task_z.unsqueeze(0).expand(self.sample_num,-1,-1).reshape(self.sample_num*t*b,-1)


        log_policy_target = torch.min(self.qf1(obs,new_actions,task_z.detach()),self.qf2(obs,new_actions,task_z.detach()))

        policy_loss = (
                self._alpha*log_pi - log_policy_target
        ).mean()
        if self.regulate_loss:
            mean_reg_loss = 0.01 * (torch.clamp(torch.abs(policy_mean)-3,min=0)).mean()
            #std_reg_loss = 0.001 * (policy_log_std**2).mean()

            policy_loss = policy_loss + mean_reg_loss #+ std_reg_loss
        regu_weight=0

    
        sparse_loss = self.agent.policy.symbolic.sparse_loss(self.sparse_ratio)
        l0_loss = self.agent.policy.symbolic.l0_loss()
        bl0_loss = self.agent.policy.symbolic.bl0_loss()

        policy_loss= policy_loss+self.spls*sparse_loss+constrain_loss*self.constrain_scale+regu_loss+self.l0_scale*l0_loss+self.bl0_scale*bl0_loss
        
 
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.grad_norm>0:
            nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_norm=self.grad_norm)
        self.policy_optimizer.step()
        with torch.no_grad():
            self.agent.policy.symbolic.proj(self.sparse_ratio)
        # end.record()
        # torch.cuda.synchronize()
        # print('policy_backward_time',start.elapsed_time(end))
        # end.record()
        # torch.cuda.synchronize()
        # print('actor_time',start.elapsed_time(end))        

        # return q gradient
        self.enable_gradients(self.qf1)
        self.enable_gradients(self.qf2)
        self.agent.policy.symbolic.batch=1
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            #self.eval_statistics['Forward DP Loss'] = np.mean(ptu.get_numpy(forward_dp_loss))
            #self.eval_statistics['Inverse DP Loss'] = np.mean(ptu.get_numpy(inverse_dp_loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Regu Loss'] = np.mean(ptu.get_numpy(regu_loss))
            self.eval_statistics['Constrain Loss'] = np.mean(ptu.get_numpy(constrain_loss))
            self.eval_statistics['Sparse Loss'] = np.mean(ptu.get_numpy(sparse_loss))

            self.eval_statistics['Sparse ratio'] = np.array(self.sparse_ratio)
            self.eval_statistics['Temp'] = np.array(self.agent.policy.symbolic.temp)
            self.eval_statistics['Count_l0'] = np.mean(ptu.get_numpy(count_l0))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            #self.eval_statistics.update(create_stats_ordered_dict(
            #    'Policy log std',
            #    ptu.get_numpy(policy_log_std),
            #))
