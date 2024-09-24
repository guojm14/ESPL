import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.sac.policies.sym_arch import get_sym_arch
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)
import math
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# TODO: deprecate classes below in favor for PolicyFromDistributionModule


def identity_regu(input):
    return input, 0
def lim_log_regu(input):
    return torch.log(torch.clamp(input,min=0.001)), torch.clamp(0.001-input,min=0).mean()
def lim_exp_regu(input):
    return torch.exp(torch.clamp(input,-10,4)),  (torch.clamp(-10-input,min=0)+torch.clamp(input-4,min=0)).mean()
def lim_second_regu(input):
    return torch.clamp(input,min=-20,max=20)**2, (torch.clamp(-20-input,min=0)+torch.clamp(input-20,min=0)).mean()
def lim_third_regu(input):
    return torch.clamp(input,min=-10,max=10)**3, (torch.clamp(-10-input,min=0)+torch.clamp(input-10,min=0)).mean()
def lim_sqrt_regu(input):
    return torch.sqrt(torch.clamp(input,min=0.0001)), (torch.clamp(0.0001-input,min=0)).mean()

def lim_div_regu(input1,input2):
    output=input1/input2.masked_fill(input2<0.01,0.01)
    output=output.masked_fill(input2<0.01,0.0)
    return output, torch.clamp(0.01-input2,min=0).mean()

def mul_regu(input1,input2):
    return torch.clamp(input1,min=-100,max=100)*torch.clamp(input2,min=-100,max=100), (torch.clamp(-100-input1,min=0)+torch.clamp(input1-100,min=0)).mean()+(torch.clamp(-100-input2,min=0)+torch.clamp(input2-100,min=0)).mean()

def cos_regu(input):
    return torch.cos(input),0
def sin_regu(input):
    return torch.sin(input),0

def identity(input):
    return input
def lim_log(input):
    return torch.log(torch.clamp(input,min=0.001))
def lim_exp(input):
    return torch.exp(torch.clamp(input,-10,4))
def lim_second(input):
    return torch.clamp(input,min=-20,max=20)**2
def lim_third(input):
    return torch.clamp(input,min=-10,max=10)**3
def lim_sqrt(input):
    return torch.sqrt(torch.clamp(input,min=0.0001))

def lim_div(input1,input2):
    output=input1/input2.masked_fill(input2<0.01,0.01)
    output=output.masked_fill(input2<0.01,0.0)
    return output

def mul(input1,input2):
    return torch.clamp(input1,min=-100,max=100)*torch.clamp(input2,min=-100,max=100)

def ifelse(condinput,input1,input2):
    cond=torch.sigmoid(condinput)
    return cond*input1+(1-cond)*input2

def ifelse_regu(condinput,input1,input2):
    cond=torch.sigmoid(condinput)
    return cond*input1+(1-cond)*input2,0
base_op=[mul,lim_div,lim_log,lim_exp,torch.sin,torch.cos,identity,ifelse]
regu_op=[mul_regu,lim_div_regu,lim_log_regu,lim_exp_regu,sin_regu,cos_regu,identity_regu,ifelse_regu]
op_in=[2,2,1,1,1,1,1,3]


# op_index_list=[]
# op_index_list.append([0,0,0,6,6,6,6,6,6])
# op_index_list.append([0,0,0])
# op_index_list.append([0,0,1,1,1])
# op_index_list.append([2,2,3,3,4,4,5,5])
# op_index_list.append([0,0,1,1,2,3])
# op_index_list.append([0,0,1,1,2,3])

op_index_list=[]
op_list=[]
op_regu_list=[]
op_in_list=[]
op_inall_list=[]

def init_op_list(index):
    global op_list
    global op_regu_list
    global op_in_list
    global op_inall_list
    global op_index_list
    op_index_list= get_sym_arch(index)
    op_list=[]
    op_regu_list=[]
    op_in_list=[]
    op_inall_list=[]    
    for layer in op_index_list:
        op=[]
        op_regu=[]
        op_in_num=[]
        op_inall=0
        for index in layer:
            op.append(base_op[index])
            op_regu.append(regu_op[index])
            op_in_num.append(op_in[index])
            op_inall+=op_in[index]
        op_list.append(op)
        op_regu_list.append(op_regu)
        op_in_list.append(op_in_num)
        op_inall_list.append(op_inall)

    print(op_in_list)
    print(op_inall_list)
    print(op_index_list)


def opfunc(input,index,mode): #input: batch,op_inall
    op_out=[]
    regu_loss=0
    if mode==1: 
        offset=0
        for i in range(len(op_in_list[index])):
            if op_in_list[index][i]==1:
                out,regu=op_regu_list[index][i](input[:,offset])
                op_out.append(out)
                regu_loss+=regu
                offset+=1
            elif op_in_list[index][i]==2:
                out,regu=op_regu_list[index][i](input[:,offset],input[:,offset+1])
                op_out.append(out)
                regu_loss+=regu
                offset+=2        
            elif op_in_list[index][i]==3:
                out,regu=op_regu_list[index][i](input[:,offset],input[:,offset+1],input[:,offset+2])
                op_out.append(out)
                regu_loss+=regu
                offset+=3   
    else:       
        offset=0
        for i in range(len(op_in_list[index])):
            if op_in_list[index][i]==1:
                out=op_list[index][i](input[:,offset])
                op_out.append(out)
                offset+=1
            elif op_in_list[index][i]==2:
                out=op_list[index][i](input[:,offset],input[:,offset+1])
                op_out.append(out)
                offset+=2  
            elif op_in_list[index][i]==3:
                out=op_list[index][i](input[:,offset],input[:,offset+1],input[:,offset+2])
                op_out.append(out)
                offset+=3
        #print(offset) 
    return torch.stack(op_out,dim=1),regu_loss

class EQL(nn.Module):
    def __init__(self, num_inputs, num_outputs,sample_num,hard_gum):
        super(EQL, self).__init__()


        self.num_inputs = num_inputs

        self.num_outputs = num_outputs

        self.hard_gum = hard_gum
        self.temp = 0.03

        
        self.repeat = 1
        self.depth=len(op_list)

        wshape=0        
        inshape_ = self.num_inputs 
        bshape=0
        for i in range(self.depth):
            bshape+=op_inall_list[i] 
            wshape+=inshape_*op_inall_list[i]	
            inshape_+=len(op_in_list[i])
        wshape+=inshape_
        bshape+=1
        self.wshape=wshape
        self.bshape=bshape
        
        self.scores= nn.Parameter(torch.Tensor(num_outputs,wshape))
        self.scores.data.fill_(1.0)
        self.constw_base = nn.Parameter(
            torch.Tensor(self.num_outputs,wshape)
        )
        
        self.constb = nn.Parameter(torch.Tensor(self.num_outputs,bshape))
        bound = 1 / math.sqrt(10)
        self.constw_base.data.uniform_(-bound, bound)
        self.constb.data.uniform_(-bound, bound)

        self.batch = 1
        self.sample_num = sample_num

    def constrain_loss(self):
        
        return torch.zeros(1).to(self.constw.device)

    def proj(self):
        self.scores.data.clamp_(0,1)
    
    def sparse_loss(self):
        clamped_scores= torch.clamp(self.scores,0,1)
        return torch.clamp(clamped_scores.sum(-1).sum(-1)-self.target_ratio*self.wshape*self.num_outputs,min=0).mean()/self.num_outputs

    # def sim_loss(self):
    #     meta_batch  = self.scores.shape[0]
    #     idx= torch.randperm(meta_batch)
    #     shuffle_scores = self.scores[idx,:,:].detach()
    #     return torch.abs(self.scores-shuffle_scores).mean()

    def score_std(self):
        return torch.std(torch.clamp(self.scores,0,1),dim=0).mean()

    def l0_loss(self):
        return torch.abs(self.constw_base).mean()

    def bl0_loss(self):
        return torch.abs(self.constb).mean()       

    def expect_w(self):
        clamped_scores= torch.clamp(self.scores,0,1)
        return clamped_scores.sum(-1).mean()

    def update_const(self):

        self.sample_sparse_constw(0)
    
    def get_loss(self):
        sparse_loss = self.sparse_loss()
        constrain_loss = self.constrain_loss()
        l0_loss = self.l0_loss()
        bl0_loss = self.bl0_loss()
        return self.spls*sparse_loss+constrain_loss*self.constrain_scale+self.regu_loss+self.l0_scale*l0_loss+self.bl0_scale*bl0_loss,sparse_loss,constrain_loss,self.regu_loss,l0_loss,bl0_loss
    def sample_sparse_constw(self,mode):                            

        if mode:
            eps = 1e-20
            scores=self.scores.unsqueeze(0).expand(self.sample_num,-1,-1)
            uniform0 = torch.rand_like(scores)
            uniform1 = torch.rand_like(scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            clamped_scores= torch.clamp(scores,0,1)
            self.constw_mask = torch.sigmoid((torch.log(clamped_scores + eps) - torch.log(1.0 - clamped_scores + eps) + noise) * self.temp)
            if self.hard_gum:
                hard_mask = torch.where(self.constw_mask>0.5, torch.ones_like(self.constw_mask),torch.zeros_like(self.constw_mask))
                constw_base= self.constw_base.unsqueeze(0).expand(self.sample_num,-1,-1)
                self.constw=constw_base*(hard_mask-self.constw_mask.detach()+self.constw_mask)
            else:
                self.constw=self.constw_base*self.constw_mask
        else:
            clamped_scores= torch.clamp(self.scores,0,1)
            self.constw_mask = (torch.rand_like(self.scores) < clamped_scores).float()
            self.constw=self.constw_base*self.constw_mask

        


    def forward(self, obs,mode=0):
        
        batch = obs.shape[0]
        x = obs
        
        
        if mode:
            self.sample_sparse_constw(1)
            #constw sample,num_outputs,wshape
            constw = self.constw.unsqueeze(1).expand(-1,batch,-1,-1).reshape(-1,self.num_outputs,self.wshape)
            constb = self.constb.unsqueeze(0).unsqueeze(1).expand(self.sample_num,batch,-1,-1).reshape(-1,self.num_outputs,self.bshape)
        else:
            constw = self.constw.unsqueeze(0).expand(batch,-1,-1)
            constb = self.constb.unsqueeze(0).expand(batch,-1,-1)


        w_list=[]
        inshape_= self.num_inputs
        low=0
        for i in range(self.depth):
            high=low+inshape_*op_inall_list[i]
            w_list.append(constw[:,:,int(low):int(high)])
            inshape_+=len(op_in_list[i])
            low=high
            
        w_last=constw[:,:,int(low):]

        b_list=[]
        low=0
        for i in range(self.depth):
            high=low+op_inall_list[i]
            b_list.append(constb[:,:,low:high])
            low=high
        b_last=constb[:,:,low:]
        #x meta_batch*batch_size,num_inputs
        if mode:
            x=x.unsqueeze(0).unsqueeze(2).expand(self.sample_num,-1,self.num_outputs,-1) #sample_num,batch,num_outputs,num_inputs
            x=x.reshape(self.sample_num*batch*self.num_outputs,self.num_inputs)
        else:
            x=x.unsqueeze(1).expand(-1,self.num_outputs,-1) #batch,num_outputs,num_inputs
            x=x.reshape(batch*self.num_outputs,self.num_inputs)
        reguloss=0
        inshape_=self.num_inputs
        if mode:
            batch = self.sample_num*batch
        for i in range(self.depth):
            w=w_list[i].reshape(batch*self.num_outputs,op_inall_list[i],inshape_)
            inshape_+=len(op_in_list[i])
            # if mode:
            #     print(b_list[i].shape,batch,self.num_outputs,op_inall_list[i])
            b=b_list[i].reshape(batch*self.num_outputs,op_inall_list[i])
            #print(w.shape,x.shape)
            hidden=torch.bmm(w, x.unsqueeze(2)).squeeze(-1)+b
            #print(hidden.shape,i,len(op_in_list[i]),op_inall_list[i])  
            op_hidden,regu=opfunc(hidden,i,mode)
            x=torch.cat([x,op_hidden],dim=-1)
            #print(x.shape)
            reguloss+=regu
        #print(self.num_inputs+(self.depth-1)*op_num*self.repeat)
        w=w_last.reshape(batch*self.num_outputs,1,inshape_)
        #print(constb.shape,b_last.shape)
        b=b_last.reshape(batch*self.num_outputs,1)
        #print(w.shape,x.shape,b.shape)
        out=torch.bmm(w, x.unsqueeze(2)).squeeze(-1)+b
        self.regu_loss = reguloss
        return out.reshape(batch,self.num_outputs)


class TanhGaussianSymbolicPolicy_V1(TorchStochasticPolicy):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            arch_index=0,
            hard_gum = True,
            sample_num = 2,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__()
        init_op_list(arch_index)
        self.sample_num = sample_num
        self.symbolic = EQL(
            obs_dim, action_dim,sample_num,hard_gum)
        self.stdnet = nn.Sequential(nn.Linear(obs_dim,64),
                                nn.ReLU(), 
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,action_dim)
        )
        self.mode=0
        self.action_dim = action_dim

    def forward(self, obs):

        mean = self.symbolic(obs,self.mode)
        log_std = self.stdnet(obs)
        if self.mode:
            log_std = log_std.unsqueeze(0).expand(self.sample_num,-1,-1).reshape(-1,self.action_dim)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)            

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

            
