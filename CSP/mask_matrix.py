import torch
import pickle
import os



def id(input):
    return input

def add(input1,input2):
    return input1+input2

def add3(input1,input2,input3):
    return input1+input2+input3

base_op=[add,add,id,id,id,id,id,add3]

op_in=[2,2,1,1,1,1,1,3]


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
    op_in_list=[]
    op_inall_list=[]    
    for layer in op_index_list:
        op=[]
        op_regu=[]
        op_in_num=[]
        op_inall=0
        for index in layer:
            op.append(base_op[index])

            op_in_num.append(op_in[index])
            op_inall+=op_in[index]
        op_list.append(op)
        op_in_list.append(op_in_num)
        op_inall_list.append(op_inall)


def opfunc(input,index): #input: batch,op_inall
    op_out=[]
  
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
    return torch.stack(op_out,dim=1)


from rlkit.torch.sac.sym_arch import get_sym_arch
def get_valid_path(num_inputs,constw):
    forward_valid= torch.zeros_like(constw)
    constw.requires_grad=True
    batch,act_dim, wshape = constw.shape
    binary_input = torch.ones(batch*act_dim,num_inputs).cuda()
    w_list=[]
    inshape_= num_inputs
    low=0
    x = binary_input
    for i in range(6):
        high=low+inshape_*op_inall_list[i]
        #print(low,high)
        this_constw = constw[:,:,int(low):int(high)]
        this_constw = this_constw.reshape(batch*act_dim,op_inall_list[i],inshape_)
        hidden = torch.bmm(this_constw, x.unsqueeze(2)).squeeze(-1)
        op_hidden=opfunc(hidden,i)
        #print(op_hidden.shape)
        this_constw = this_constw*x.unsqueeze(-2)
        forward_valid[:,:,int(low):int(high)] = this_constw.reshape(batch, act_dim, -1)
        x = torch.cat([x,op_hidden],dim=-1)
        inshape_+=len(op_in_list[i])
        low=high
    w_last=constw[:,:,int(low):]
    w_last = w_last.reshape(batch*act_dim,1,wshape-int(low))
    w_last=w_last*x.unsqueeze(-2)
    forward_valid[:,:,int(low):] = w_last.reshape(batch, act_dim, -1)
    out = torch.bmm(w_last, x.unsqueeze(2)).squeeze(-1)
    
    out.mean().backward()

    #print(constw.grad.data.shape)
    return forward_valid*(torch.abs(constw.grad.data)>0).float()


dir=''
constwmask='.pkl'
init_op_list(0)
with open(os.path.join(dir,constwmask),'rb') as f:
    mask=(pickle.load(f)[0]>0.00001).float()
    valid_mask=(get_valid_path(11,mask)>0.00001).float()
    most9_mask=(valid_mask.mean(0)>=0.9).float().sum(-1)
    print('average_mask',valid_mask.sum(-1).mean())
    print('most_mask',most9_mask.mean())

