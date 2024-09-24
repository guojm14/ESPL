from tkinter import EW
import sympy
import torch
from rlkit.torch.sac.sym_arch import get_sym_arch
def sym_matmul(x,weight,bias): #weight tensor out, in
    exp_list=[]
    for i in range(weight.shape[0]):
        exp = 0
        for j in range(weight.shape[1]):
            exp += x[j]*weight[i,j].item()
        exp += bias[i]
        exp_list.append(sympy.expand(exp))
    return exp_list

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

def mul(x1,x2): 
    return x1*x2
def div(x1,x2):
    return x1/x2
def identity(x):
    return x
def ifelse(x1,x2,x3):
    cond=1/(1 + sympy.exp(-x1))
    return cond*x2+(1-cond)*x3
base_op = [mul,div,sympy.log,sympy.exp,sympy.sin,sympy.cos,identity,ifelse]
op_in=[2,2,1,1,1,1,1,3]


# for layer in op_index_list:
#     op=[]
#     op_in_num=[]
#     op_inall=0
#     for index in layer:
#         op.append(base_op[index])
#         op_in_num.append(op_in[index])
#         op_inall+=op_in[index]
#     op_list.append(op)
#     op_in_list.append(op_in_num)
#     op_inall_list.append(op_inall)

op_index_list=[]
op_list=[]
op_in_list=[]
op_inall_list=[]

def print_init_op_list(index):
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
            op_out.append(sympy.expand(out))
            offset+=1
        elif op_in_list[index][i]==2:
            #print(op_list[index][i])
            out=op_list[index][i](input[offset],input[offset+1])
            op_out.append(sympy.expand(out))
            offset+=2
        elif op_in_list[index][i]==3:
            #print(op_list[index][i])
            out=op_list[index][i](input[offset],input[offset+1],input[offset+2])
            op_out.append(sympy.expand(out))
            offset+=3
    return op_out



# def printwymbolic(constw,num_inputs):
#     vars = sympy.symbols('x:'+str(num_inputs))
#     for i in range(constw.shape[0]):

# a=torch.randn(90).reshape(9,10)/10
# a=torch.where(torch.abs(a)>0.1,a,torch.zeros_like(a))
# vars=sympy.symbols('x:10')

# exp=opfunc(sym_matmul(vars,a),1)
# print(exp)


import timeout_decorator
@timeout_decorator.timeout(60)
def printsymbolic(constw,constb,num_inputs,latent_dim,depth):
    print('start')
    #constw=constw/10
    constw=torch.where(torch.abs(constw)>0.01,constw,torch.zeros_like(constw))
    x = sympy.symbols('x:'+str(num_inputs-latent_dim))
    x =list(x)
    #print(constw)
    for j in range(constw.shape[0]):

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


        inshape_=num_inputs-latent_dim
        for i in range(depth):
            w=w_list[i].view(op_inall_list[i],inshape_)
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
        #print(constb.shape,b_last.shape)
        b=b_last.view(1)
        #print(w.shape,x.shape,b.shape)
        out=sym_matmul(x,w,b)
        #print(x,w,out)
        print(round_expr(sympy.expand(out[0]),2))
'''
num_inputs=3
latent_dim=1
wshape=0
depth=len(op_list)
inshape_ = num_inputs - latent_dim
bshape=0
for i in range(depth):
    bshape+=op_inall_list[i]
    wshape+=inshape_*op_inall_list[i]
    inshape_+=len(op_in_list[i])
wshape+=inshape_
bshape+=1

constw=torch.randn(2*wshape).reshape(2,-1)/10
constw=torch.where(torch.abs(constw)>0.1,constw,torch.zeros_like(constw))
constb=torch.randn(2*bshape).reshape(2,-1)
#print(constw,constb)
printsymbolic(constw,constb,num_inputs,latent_dim,depth)
'''
