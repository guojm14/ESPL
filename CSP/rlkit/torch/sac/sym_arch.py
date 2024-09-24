op_list=[]

op_index_list=[]
op_index_list.append([0,0,0])
op_index_list.append([0,0,0])
op_index_list.append([1,1,1])
op_index_list.append([2,3,4,4,4,5,5,5])
op_index_list.append([0,1,2,3])
op_index_list.append([0,1,2,3])
op_list.append(op_index_list)



op_index_list=[]
op_index_list.append([0,0,0,7])
op_index_list.append([0,0,0,7])
op_index_list.append([1,1,1,7])
op_index_list.append([2,3,4,4,4,5,5,5,7])
op_index_list.append([0,1,2,3,7])
op_list.append(op_index_list)


def get_sym_arch(index):
    return op_list[index]

