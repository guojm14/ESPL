op_list=[]

op_index_list=[]
op_index_list.append([0,0,0])
op_index_list.append([1,1,1])
op_index_list.append([2,2,3,3,4,4,5,5])
op_index_list.append([2,2,3,3,4,4,5,5])
op_index_list.append([0,1,2,3])
op_index_list.append([0,1,2,3])
op_list.append(op_index_list)



def get_sym_arch(index):
    return op_list[index]

