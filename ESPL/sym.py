from tkinter import EW
import sympy
import torch
from rlkit.torch.sac.policies.sym_arch import get_sym_arch
import pickle
def sym_matmul(x,weight,bias): #weight tensor out, in
    exp_list=[]
    for i in range(weight.shape[0]):
        exp = 0
        for j in range(weight.shape[1]):
            exp += x[j]*weight[i,j].item()
        exp += bias[i]
        exp_list.append(exp)
    return exp_list

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

def mul(x1,x2): 
    return x1*x2
def div(x1,x2):
    return x1/x2
def identity(x):
    return x

ifelse=sympy.Function('ifelse')
base_op = [mul,div,sympy.log,sympy.exp,sympy.sin,sympy.cos,identity,ifelse]
op_in=[2,2,1,1,1,1,1,3]


op_index_list=[]
op_list=[]
op_in_list=[]
op_inall_list=[]

def init_op_list(index):
    global op_list
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
            out=op_list[index][i](input[offset])
            op_out.append(out)
            offset+=1
        elif op_in_list[index][i]==2:
            #print(op_list[index][i])
            out=op_list[index][i](input[offset],input[offset+1])
            op_out.append(out)
            offset+=2
        elif op_in_list[index][i]==3:
            #print(op_list[index][i])
            out=op_list[index][i](input[offset],input[offset+1],input[offset+2])
            op_out.append(out)
            offset+=3
    return op_out



import timeout_decorator
@timeout_decorator.timeout(60)
def printsymbolic(constw,constb,num_inputs,arch_index):
    init_op_list(arch_index)
    depth=len(op_list)
    print('start')
    #constw=constw/10
    constw=torch.where(torch.abs(constw)>0.001,constw,torch.zeros_like(constw))
    #print(constw)
    for j in range(constw.shape[0]):
        # print((torch.abs(constw[j])>0).float().sum())
        x = sympy.symbols('x:'+str(num_inputs))
        x =list(x)
        w_list=[]
        inshape_= num_inputs
        low=0
        for i in range(depth):
            high=low+inshape_*op_inall_list[i]
            w_list.append(constw[j,int(low):int(high)])
            inshape_+=len(op_in_list[i])
            low=high

        w_last=constw[j,int(low):]

        b_list=[]
        low=0
        for i in range(depth):
            high=low+op_inall_list[i]
            b_list.append(constb[j,low:high])
            low=high
        b_last=constb[j,low:]
        # for item in w_list:
        #     print(item.shape)

        inshape_=num_inputs
        
        for i in range(depth):
            w=w_list[i].view(op_inall_list[i],inshape_)
            inshape_+=len(op_in_list[i])
            b=b_list[i].view(op_inall_list[i])
            hidden=sym_matmul(x,w,b)
            op_hidden=opfunc(hidden,i)
            x=x+op_hidden

        w=w_last.view(1,inshape_)
        b=b_last.view(1)
        out=sym_matmul(x,w,b)
        print(round_expr(out[0],2))
  
import argparse
import os
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Symbolic')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--arch_index', type=int, default=1)
    parser.add_argument('--num_inputs', type=int, default=24) #hopper 15 cheetah 25 lunar 
    args = parser.parse_args()
    model=torch.load(args.path)
    scores=model['trainer/policy'].symbolic.scores.data
    constw_base= model['trainer/policy'].symbolic.constw_base.data
    constb= model['trainer/policy'].symbolic.constb.data
    constw=constw_base*((scores>0.5).float())
    init_op_list(args.arch_index)
    print(torch.abs(scores-0.5).mean())
    print(constw_base.shape,scores.shape)
    depth=len(op_list)
    print(depth)
    printsymbolic(constw,constb,args.num_inputs,args.arch_index)
