import numpy as np
from numpy import linalg as LA
import numpy.matlib

CPU_THRESHOLD =0
LINK_THRESHOLD=0

def get_hungarian_matching(M) :
    P=np.zeros(np.shape(M))
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(M)
    P[row_ind, col_ind] = 1
    return P, row_ind, col_ind

def get_remaining_elements(arr, removeIndx) :
    mask = np.ones(arr.shape[1], dtype=bool)
    mask[removeIndx] = False
    return arr[0,mask]

def get_non_zero_min(arr):
    return np.min(arr[np.nonzero(arr)])

def do_cpu_check(M, A_sg, A_rg, num_request_nodes):
    cpu_M=M
    #rows, cols =  np.shape(A_rg[np.any(A_rg, axis=1)])
    rows, cols=np.shape(M)
    for j in range(num_request_nodes):
        for k in range(cols):
            sg_max_bw= np.max(get_remaining_elements(A_sg[k],k))
            rg_min_bw=get_non_zero_min(get_remaining_elements(A_rg[j],j))
            if A_sg[k,k] - A_rg[j,j] < CPU_THRESHOLD :
                #check CPU constraints
                cpu_M[j,k]=float("inf")

            elif np.max(get_remaining_elements(A_sg[k],k)) - \
                    get_non_zero_min(get_remaining_elements(A_rg[j],j)) < LINK_THRESHOLD :
                # check Link constraints
                # If maximum bw of all the links of a substrate node
                # is still less than the minimum link bw of request node
                # then the mapping cannot be possible
                cpu_M[j, k] = float("inf")


    return cpu_M

def get_VRE_mapping(A_sg, R_rg):

    num_request_nodes = R_rg.shape[0]
    A_rg=np.matlib.zeros(A_sg.shape)
    A_rg[:num_request_nodes, :num_request_nodes] = R_rg

    #Compute eigen values
    w_sg, U_sg = LA.eig(A_sg)
    w_rg, U_rg = LA.eig(A_rg)

    abs_U_sg = np.absolute(U_sg)
    abs_U_rg = np.absolute(U_rg)
    M = abs_U_rg * (abs_U_sg.T)
    cpu_M = do_cpu_check(M=M, A_sg=A_sg, A_rg=A_rg, num_request_nodes=num_request_nodes)
    P, row_ind, col_ind = get_hungarian_matching(cpu_M)


    '''
    print M
    print cpu_M
    print U_sg
    print w_sg
    print U_rg
    print abs_U_sg
    print abs_U_rg
    '''

    return P, row_ind, col_ind

if  __name__ =='__main__' :

    A_sg = np.matrix('3 7 7 3 1 5; '
                     '7 8 9  2  3  5 ;'
                     '7 9 15  4  10 5 ; '
                     '3 2 4  5  3 4 ; '
                     '1 3 10 3 11 5 ;  '
                     '5 5 5  4 5 2')

    R_rg = np.matrix(' 8 5 2 ;5 1 0; 2 0 3 ')

    P, row_ind, col_ind = get_VRE_mapping(A_sg=A_sg, R_rg=R_rg)
    rg_nodes = map(list,zip(*[iter(row_ind)]*1))
    sg_nodes = map(list,zip(*[iter(col_ind)]*1))
    #print rg_nodes
    #print sg_nodes
    mapping=np.hstack((rg_nodes, sg_nodes))
    print mapping[:R_rg.shape[0]]
    #print P
