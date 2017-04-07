import numpy as np
from numpy import linalg as LA
import numpy.matlib
import constants as cs

CPU_THRESHOLD =0
LINK_THRESHOLD=0
DIAGONAL_CONSTANT=1

class InvalidMappingRequest(Exception):
   """Raised when the request/Substrate graph has any unconnected nodes"""
   pass

class UnacceptableRequest(Exception):
   """Raised when the request graph has requirements that cannot be satisfied due to their exceeding capacities"""
   pass

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
    try:
        min_val = np.min(arr[np.nonzero(arr)])
    except Exception :
        raise InvalidMappingRequest
    return min_val

def do_cpu_check(M, A_sg, A_rg, num_request_nodes):
    cpu_M=M
    #rows, cols =  np.shape(A_rg[np.any(A_rg, axis=1)])
    rows, cols=np.shape(M)
    for j in range(num_request_nodes):
        for k in range(cols):
            try:
                #sg_max_bw= np.max(get_remaining_elements(A_sg[k],k))
                #rg_min_bw=get_non_zero_min(get_remaining_elements(A_rg[j],j))
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
            except InvalidMappingRequest :
                raise InvalidMappingRequest("Invalid request with no connectivity for node ", j+1)

    return cpu_M

def get_my_VRE_mapping(A_sg, R_rg):
    #Consider only bandwidths for assignment. So we assign all cpu_capacities captured in diagonal
    #elements as a constant value DIAGONAL_CONSTANT
    #np.fill_diagonal(A_sg, DIAGONAL_CONSTANT)

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
    if is_acceptable(M) :
        P, row_ind, col_ind = get_hungarian_matching(cpu_M)

    '''
    print w_sg
    print w_rg

    print M
    print cpu_M
    print U_sg
    print U_rg
    print abs_U_sg
    print abs_U_rg
    '''

    return P, row_ind, col_ind

def is_acceptable(M) :
    mask = np.all(np.isinf(M) | np.equal(M, 0), axis=1)
    if np.any(mask) :
        raise UnacceptableRequest( "Request cannot be satisfied as the following request nodes cannot be mapped: ",
                                   list(np.argwhere(mask)[:,0]+1))

    return  True


def fill_substrate_matrix(A_sg):
    import networkx as nx
    import networkx_mst_src as nx_loc

    G = nx.from_numpy_matrix(A_sg)

    T = nx_loc.maximum_spanning_tree(G)
    #print sorted(T.edges(data=True))

    #non_edges = np.argwhere(np.isnan(A_sg))
    non_edges = np.argwhere(A_sg==0)
    for ne in non_edges:
        p = list(nx.shortest_simple_paths(T, ne[0], ne[1]))[0]
        min_val = float("inf")
        for i in range(len(p) - 1):
            temp = T.get_edge_data(p[i], p[i + 1])['weight']
            if temp < min_val:
                min_val=temp
        A_sg[ne[0],ne[1]] = min_val

    return A_sg

def get_graph_matrices(dir_name):
    A_sg = np.matrix(np.genfromtxt(cs.DATA_PATH+dir_name+'/'+cs.SG_FILE_NAME, delimiter=',', dtype=float))
    R_rg = np.matrix(np.genfromtxt(cs.DATA_PATH+dir_name+'/'+cs.RG_FILE_NAME, delimiter=',', dtype=float))

    # nan means no edges between the nodes, we replace it by 0
    A_sg[np.isnan(A_sg)]=0
    R_rg[np.isnan(R_rg)] = 0

    return fill_substrate_matrix(A_sg), R_rg

def compute_mapping(A_sg, R_rg):
    P, row_ind, col_ind = get_my_VRE_mapping(A_sg=A_sg, R_rg=R_rg)
    rg_nodes = map(list, zip(*[iter(row_ind)] * 1))
    sg_nodes = map(list, zip(*[iter(col_ind)] * 1))
    # print rg_nodes
    # print sg_nodes
    # print P
    mapping = np.hstack((rg_nodes, sg_nodes))
    print mapping[:R_rg.shape[0]]

    return mapping

def get_mapping(dir_name):
    try:
        A_sg, R_rg = get_graph_matrices(dir_name=dir_name)
        compute_mapping(A_sg, R_rg)
    except Exception as error:
        print('Caught this error: ' + repr(error))

def test_mapping() :
    A_sg_init = np.matrix('3 7 0 3 1 0; '
                     '7 8 9 2 0 0 ;'
                     '0 9 15 4 10 0 ; '
                     '3 2 4 5 3 0; '
                     '1 0 10 3 11 5 ;  '
                     '0 0 0 0 5 2')

    A_sg=fill_substrate_matrix(A_sg_init)

    '''
    #Matrix from paper - after filling
    A_sg_paper = np.matrix('3 7 7 3 1 5; '
                     '7 8 9  2  3  5 ;'
                     '7 9 15  4  10 5 ; '
                     '3 2 4  5  3 4 ; '
                     '1 3 10 3 11 5 ;  '
                     '5 5 5  4 5 2')
    '''
    #print np.all(A_sg, A_sg_paper)
    R_rg = np.matrix(' 8 5 2 ;5 1 0; 2 0 3 ')
    compute_mapping(A_sg, R_rg)

if  __name__ =='__main__' :
    #test_mapping()
    get_mapping('sg_20_rg_15')
