"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
import copy
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
import numpy as np

def identity(x):
    return x



class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,deterministic=True,tanh=False):
        super(Model,self).__init__()
        self.main=nn.Sequential(nn.Linear(input_size,hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size,hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size,hidden_size),
                  nn.ReLU(),
                  )
        self.deterministic= deterministic
        self.tanh=tanh
        if deterministic:
            self.output=nn.Linear(hidden_size,output_size)
        else:
            self.mu=nn.Linear(hidden_size,output_size)
            self.std=nn.Linear(hidden_size,output_size)
        
    def forward(self, *input):
        flat_input=torch.cat(input,dim=1)
        if self.deterministic:
            if self.tanh:
                return F.tanh(self.output(self.main(flat_input)))
            else:
                return self.output(self.main(flat_input))
        else:
            if self.tanh:
                mu = F.tanh(self.mu(self.main(input)))
            else:
                mu = self.mu(self.main(input))
            std=F.softplus(self.main(flat_input))
            dist=torch.distributions.Normal(mu,std)
            return dist.rsample()
               

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            obs_dim,
            latent_dim,
            action_dim,
            input_size,
            init_w=3e-3,
            z_dim=100,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0., #0.1
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.latent_dim = latent_dim
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0) #uniform

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class MultiLinear(nn.Module):
  def __init__(self, in_channels, out_channels, num_linears=2,b_init_value=0.1):
    super().__init__()
    self.W = torch.nn.Parameter(torch.randn(num_linears, in_channels, out_channels))
    self.b = torch.nn.Parameter(torch.zeros(num_linears, out_channels))
    self.b.data.fill_(b_init_value)
    bound = 1 / np.sqrt(in_channels) 
    self.W.data.uniform_(-bound, bound)

  def forward(self, x, id):
    #print(id)
    #print(self.W.shape,x.shape,self.W[id].shape)
    out = torch.bmm(x, self.W[id])     # (t,b,out_c)
    out = out + self.b[id].unsqueeze(1) 
    return out

class EnsembleMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            obs_dim,
            latent_dim,
            action_dim,
            input_size,
            use_reverse,
            use_combine,
            init_w=3e-3,
            z_dim=100,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.latent_dim = latent_dim
        self.fcs = []
        self.layer_norms = []
        in_size = obs_dim+action_dim
        for i, next_size in enumerate(hidden_sizes):
            fc = MultiLinear(in_size, next_size,40)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
        


        self.last_fc = MultiLinear(in_size, output_size,40)
        self.last_fc.W.data.uniform_(-init_w, init_w)
        self.last_fc.b.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False,id=None):
        t=len(id)
        b=input.shape[0]//t
        #print(input.shape,t,b)
        id=torch.tensor(id).to(input.device)
        h = input.view(t,b,-1)
        for i, fc in enumerate(self.fcs):
            h = fc(h,id)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h,id)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output.reshape(t*b,-1), preactivation.reshape(t*b,-1)
        else:
            return output.reshape(t*b,-1)


class SimpleMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()


        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc_log_std = nn.Linear(last_hidden_size, output_size)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean,std

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        if len(inputs) > 1 and inputs[1] is None:
            flat_inputs = torch.cat([inputs[0],inputs[2]], dim=1)
        else:
            flat_inputs = torch.cat(inputs, dim=1)
                

        return super().forward(flat_inputs, **kwargs)

class FlattenEnsembleMlp(EnsembleMlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        if len(inputs) > 1 and inputs[1] is None:
            flat_inputs = torch.cat([inputs[0],inputs[2]], dim=1)
        else:
            flat_inputs = torch.cat(inputs, dim=1)
                

        return super().forward(flat_inputs, **kwargs)
       
class EmbeddingMLP(Mlp):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.z_embedding = nn.Sequential(nn.Linear(self.latent_dim,32),
                                        nn.ReLU(), 
                                        nn.Linear(32,5))
    def forward(self, obs,action,task_z,**kwargs):
        z=self.z_embedding(task_z)
        flat_inputs = torch.cat([obs,action,z],dim=1)
        return super().forward(flat_inputs, **kwargs)


    

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)




