from random import shuffle
from torch.autograd.function import Function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.core import np_ify
from rlkit.torch.sac.sym import printsymbolic,print_init_op_list

from rlkit.torch.sac.sym_arch import get_sym_arch

def sym_matmul(x,weight): #weight tensor out, in
    exp_list=[]
    for i in range(weight.shape[0]):
        exp = 0
        for j in range(weight.shape[1]):
            exp += x[j]*weight[i,j].item()
        exp_list.append(exp)
    return exp


        
        

class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.Tensor(self.num_members, self.in_size, self.out_size)
        )
        
        self.bias = nn.Parameter(torch.Tensor(self.num_members, 1, self.out_size))
        bound = 1 / math.sqrt(self.in_size)
        self.weight.data.uniform_(-bound, bound)
        self.bias.data.uniform_(-bound, bound)

    def forward(self, x): # x: ...,n,1,d_in
        xw = x.matmul(self.weight) # ...,n,1,d_out
        return xw + self.bias # ...,n,1,d_out

class EnsembleMLP(nn.Module):
    def __init__(self,num_inputs,hidden_size,layer_num,output_size,num_members):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_num = layer_num
        if self.layer_num>1:
            self.layers.append(EnsembleLinearLayer(num_members,num_inputs,hidden_size))
            for i in range(layer_num-2):
                self.layers.append(EnsembleLinearLayer(num_members,hidden_size,hidden_size))
            self.layers.append(EnsembleLinearLayer(num_members,hidden_size,output_size))
        else:
            self.layers.append(EnsembleLinearLayer(num_members,num_inputs,output_size))
    def forward(self,x):

        x=x.unsqueeze(1).unsqueeze(-2)
        layer_count=0
        for layer in self.layers:
            x = layer(x)
            layer_count+=1
            if not layer_count==self.layer_num:
                x=F.relu(x)
        x=x.squeeze(-2)
        return x
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
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
    print_init_op_list(index)
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
    def __init__(self, num_inputs, latent_dim,repeat, depth, num_outputs,context_hidden_dim,sample_num,hard_gum):
        super(EQL, self).__init__()


        self.num_inputs = num_inputs
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs

        self.hard_gum = hard_gum
        self.temp = 0.03

        
        self.repeat = 1
        self.depth=len(op_list)

        wshape=0        
        inshape_ = self.num_inputs - self.latent_dim
        bshape=0
        for i in range(self.depth):
            bshape+=op_inall_list[i] 
            wshape+=inshape_*op_inall_list[i]	
            inshape_+=len(op_in_list[i])
        wshape+=inshape_
        bshape+=1
        self.wshape=wshape
        self.bshape=bshape
        self.latent2constw=EnsembleMLP(latent_dim,context_hidden_dim,3,wshape,num_outputs)
        self.latent2score = EnsembleMLP(latent_dim,context_hidden_dim,3,wshape,num_outputs)
        self.latent2score.layers[-1].bias.data.fill_(3)
        self.latent2constb=EnsembleMLP(latent_dim,context_hidden_dim,3,bshape,num_outputs)
        self.batch = 1
        self.test=False
        self.test_mode = 0
        self.sample_num = sample_num
    
    def printconstw(self,index):
        printsymbolic(self.constw[index],self.constb[index],self.num_inputs,self.latent_dim,self.depth)

    def constrain_loss(self):
        
        return torch.clamp(self.scores-6,min=0).sum(-1).sum(-1).mean()+torch.clamp(-6-self.scores,min=0).sum(-1).sum(-1).mean()


    
    def sparse_loss(self,target_ratio):
        clamped_scores= torch.sigmoid(self.scores)
        return torch.clamp(clamped_scores.sum(-1).sum(-1)-target_ratio*self.wshape*self.num_outputs,min=0).mean()/self.num_outputs




    def l0_loss(self):
        return torch.abs(self.constw_base).mean()

    def bl0_loss(self):
        return torch.abs(self.constb).mean()       

    def expect_w(self):
        clamped_scores= torch.sigmoid(self.scores)
        return clamped_scores.sum(-1).mean()

    def proj(self,sparse_ratio):
        pass

    def update_const(self,latent):
        latent=latent.to(self.latent2score.layers[-1].weight.data.device).detach()
        self.constw_base=self.latent2constw(latent)
        self.constb=self.latent2constb(latent)
        self.scores=self.latent2score(latent)
        self.sample_sparse_constw(0)
    
    def sample_sparse_constw(self,mode):                            

        if mode:
            eps = 1e-20
            scores=self.scores.unsqueeze(0).expand(self.sample_num,-1,-1,-1)
            uniform0 = torch.rand_like(scores)
            uniform1 = torch.rand_like(scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            clamped_scores= torch.sigmoid(scores)
            self.constw_mask = torch.sigmoid((torch.log(clamped_scores + eps) - torch.log(1.0 - clamped_scores + eps) + noise) * self.temp)
            if self.hard_gum:
                hard_mask = torch.where(self.constw_mask>0.5, torch.ones_like(self.constw_mask),torch.zeros_like(self.constw_mask))
                constw_base= self.constw_base.unsqueeze(0).expand(self.sample_num,-1,-1,-1)
                self.constw=constw_base*(hard_mask-self.constw_mask.detach()+self.constw_mask)
            else:
                self.constw=self.constw_base*self.constw_mask
        else:
            clamped_scores= torch.sigmoid(self.scores)
            self.constw_mask = (torch.rand_like(self.scores) < clamped_scores).float()
            self.constw=self.constw_base*self.constw_mask

        


    def forward(self, obs,mode=0):
        
        batch = obs.shape[0]
        x = obs[:,:-self.latent_dim]
        
        
        if mode:
            self.sample_sparse_constw(1)
            #constw sample,meta_batch,num_outputs,wshape
            constw = self.constw.unsqueeze(2).expand(-1,-1,self.batch,-1,-1).reshape(-1,self.num_outputs,self.wshape)
            constb = self.constb.unsqueeze(0).unsqueeze(2).expand(self.sample_num,-1,self.batch,-1,-1).reshape(-1,self.num_outputs,self.bshape)
        else:
            constw = self.constw.unsqueeze(1).expand(-1,self.batch,-1,-1).reshape(batch,self.num_outputs,-1)
            constb = self.constb.unsqueeze(1).expand(-1,self.batch,-1,-1).reshape(batch,self.num_outputs,-1)


        w_list=[]
        inshape_= self.num_inputs - self.latent_dim
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
            x=x.reshape(self.sample_num*batch*self.num_outputs,self.num_inputs-self.latent_dim)
        else:
            x=x.unsqueeze(1).expand(-1,self.num_outputs,-1) #batch,num_outputs,num_inputs
            x=x.reshape(batch*self.num_outputs,self.num_inputs-self.latent_dim)
        reguloss=0
        inshape_=self.num_inputs-self.latent_dim
        if mode:
            batch = self.sample_num*batch
        for i in range(self.depth):
            w=w_list[i].view(batch*self.num_outputs,op_inall_list[i],inshape_)
            inshape_+=len(op_in_list[i])
            #print(b_list[i].shape)
            b=b_list[i].view(batch*self.num_outputs,op_inall_list[i])
            #print(w.shape,x.shape)
            hidden=torch.bmm(w, x.unsqueeze(2)).squeeze(-1)+b
            #print(hidden.shape,i,len(op_in_list[i]),op_inall_list[i])  
            op_hidden,regu=opfunc(hidden,i,mode)
            x=torch.cat([x,op_hidden],dim=-1)
            #print(x.shape)
            reguloss+=regu
        #print(self.num_inputs+(self.depth-1)*op_num*self.repeat)
        w=w_last.view(batch*self.num_outputs,1,inshape_)
        #print(constb.shape,b_last.shape)
        b=b_last.view(batch*self.num_outputs,1)
        #print(w.shape,x.shape,b.shape)
        out=torch.bmm(w, x.unsqueeze(2)).squeeze(-1)+b
        return out.reshape(batch,self.num_outputs),reguloss




MU_MIN = -1e10
MU_MAX = 1e10



class TanhGaussianSymbolic(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            latent_dim,
            input_size,
            std=None,
            init_w=1e-3,
            repeat=2,
            depth=3,
            std_hidden_dim=64,
            context_hidden_dim=64,
            arch_index=1,
            hard_gum = False,
            sample_num=2,
            **kwargs
    ):
        super().__init__()
        self.depth = depth
        #print(input_size, latent_dim, 2,3,action_dim)
        init_op_list(arch_index)
        self.symbolic = EQL(
            input_size, latent_dim, repeat,depth,action_dim,context_hidden_dim,sample_num,hard_gum)
        self.stdnet = nn.Sequential(nn.Linear(input_size,std_hidden_dim),
                                nn.ReLU(), 
                                nn.Linear(std_hidden_dim,std_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(std_hidden_dim,std_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(std_hidden_dim,action_dim)
        )
        self.pointer = 0
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.sample_num = sample_num
        self.mode = 0
    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        
        mean,self.regu_loss = self.symbolic(obs,self.mode)
        
        log_std = self.stdnet(obs)
        if self.mode:
            log_std = log_std.unsqueeze(0).expand(self.sample_num,-1,-1).reshape(-1,self.action_dim)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        #print(mean)
        mean = torch.clamp(mean, MU_MIN, MU_MAX)
        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            try:
                tanh_normal = TanhNormal(mean, std)
            except:
                print('mean',mean)
                print('loga',self.symbolic.constw_mask.sum().isnan(),self.symbolic.constw_mask.sum().isinf())
                print('constw',self.symbolic.constw.sum().isnan(),self.symbolic.constw.sum().isinf())
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)


