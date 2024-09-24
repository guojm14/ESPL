from tkinter import EW
import sympy
import torch
from rlkit.torch.sac.sym_arch import get_sym_arch
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
def printsymbolic(constw,constb,num_inputs,latent_dim,depth):
    print('start')

    x = sympy.symbols('x:'+str(num_inputs-latent_dim))
    x =list(x)

    for j in range(constw.shape[0]):
        print((torch.abs(constw[j])>0).float().sum())
        x = sympy.symbols('x:'+str(num_inputs-latent_dim))
        x =list(x)
        w_list=[]
        inshape_= num_inputs - latent_dim
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
        for item in w_list:
            print(item.shape)

        inshape_=num_inputs-latent_dim
        
        for i in range(depth):
            w=w_list[i].view(op_inall_list[i],inshape_)
            #print(w)
            inshape_+=len(op_in_list[i])
            #print(b_list[i].shape)
            b=b_list[i].view(op_inall_list[i])
            #print(w.shape,x.shape)
            hidden=sym_matmul(x,w,b)
            #print(hidden.shape,i,len(op_in_list[i]),op_inall_list[i])
            op_hidden=opfunc(hidden,i)
            x=x+op_hidden
            #print(x.shape)
           
        #print(self.num_inputs+(self.depth-1)*op_num*self.repeat)
        w=w_last.view(1,inshape_)
        #print(w_last)
        #print(constb.shape,b_last.shape)
        b=b_last.view(1)
        #print(w.shape,x.shape,b.shape)
        out=sym_matmul(x,w,b)
        #print(x,w,out)
        print(round_expr(out[0],2))
import argparse
import os
from rlkit.envs import ENVS
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Symbolic')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--pathw', type=str, default=None)
    parser.add_argument('--pathb', type=str, default=None)
    parser.add_argument('--arch_index', type=int, default=1)
    parser.add_argument('--meta_index', type=int, default=0)
    parser.add_argument('--num_inputs', type=int, default=9) 
    parser.add_argument('--latent_dim', type=int, default=5)
    args = parser.parse_args()
    pathw=os.path.join(args.path,args.pathw)
    pathb= os.path.join(args.path,args.pathb)
    fopw=open(pathw,'rb')
    fopb=open(pathb,'rb')
    constw=pickle.load(fopw).unsqueeze(-2)
    print(constw.shape)
    constw=constw[0,args.meta_index,:,:]
    constb=pickle.load(fopb)
    print(constw.shape,constb.shape)
    constb=constb[args.meta_index,:]
    fopw.close()
    fopb.close()
    init_op_list(args.arch_index)
    depth=len(op_list)
    printsymbolic(constw,constb,args.num_inputs,args.latent_dim,depth)

