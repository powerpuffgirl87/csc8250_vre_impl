import numpy as np
from numpy import linalg as LA
import numpy.matlib
import networkx as nx
import networkx_mst_src as nx_loc
import pickle

import constants as cs
import warnings
warnings.filterwarnings("error")


CPU_THRESHOLD =0
LINK_THRESHOLD=0
DIAGONAL_CONSTANT=1
NO_MAPPING_VALUE=np.inf
#MAX_VALUE=1000000000
MAX_VALUE=np.inf
NO_LINK_VALUE=0

class InvalidMappingRequest(Exception):
   """Raised when the request/Substrate graph has any unconnected nodes"""
   pass

class UnacceptableRequest(Exception):
   """Raised when the request graph has requirements that cannot be satisfied due to their exceeding capacities"""
   pass

def get_hungarian_matching(M) :
    P=np.zeros(np.shape(M))
    from scipy.optimize import linear_sum_assignment
    try:
        row_ind, col_ind = linear_sum_assignment(M)
        P[row_ind, col_ind] = 1
        return P, row_ind, col_ind
    except RuntimeWarning:
        raise UnacceptableRequest("Request cannot be mapped")

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
            subs_bws=get_remaining_elements(A_sg[k],k)
            req_bws=get_remaining_elements(A_rg[j],j)
            try:
                #sg_max_bw= np.max(get_remaining_elements(A_sg[k],k))
                #rg_min_bw=get_non_zero_min(get_remaining_elements(A_rg[j],j))
                if A_sg[k,k] - A_rg[j,j] < CPU_THRESHOLD :
                    #check CPU constraints
                    cpu_M[j,k]=NO_MAPPING_VALUE

                elif np.max(subs_bws) - get_non_zero_min(req_bws) < LINK_THRESHOLD or \
                     np.max(subs_bws) - np.max(req_bws) < LINK_THRESHOLD or \
                     np.sum(subs_bws) - np.sum(req_bws) < LINK_THRESHOLD :
                    # check Link constraints
                    # If maximum bw of all the links of a substrate node
                    # is still less than the minimum  or maximum link bw of request node (or)
                    # sum of all outgoing bandwith of substrate node < to that of request node
                    # then the mapping cannot be possible
                    cpu_M[j, k] = NO_MAPPING_VALUE
            except InvalidMappingRequest :
                raise InvalidMappingRequest("Invalid request with no connectivity for node ", j+1)

    return cpu_M
def get_eigen_matching(A_sg, R_rg) :
    # Consider only bandwidths for assignment. So we assign all cpu_capacities captured in diagonal
    # elements as a constant value DIAGONAL_CONSTANT
    A_sg_new, fill_tree = fill_substrate_matrix(np.matrix(A_sg))
    # np.fill_diagonal(A_sg_new, DIAGONAL_CONSTANT)

    num_request_nodes = R_rg.shape[0]
    A_rg = np.matlib.zeros(A_sg_new.shape)
    A_rg[:num_request_nodes, :num_request_nodes] = R_rg

    # Compute eigen values
    w_sg, U_sg = LA.eig(A_sg_new)
    w_rg, U_rg = LA.eig(A_rg)

    abs_U_sg = np.absolute(U_sg)
    abs_U_rg = np.absolute(U_rg)
    M = abs_U_rg * (abs_U_sg.T)
    cpu_M = do_cpu_check(M=M, A_sg=A_sg_new, A_rg=A_rg, num_request_nodes=num_request_nodes)
    return cpu_M, fill_tree

def get_my_VRE_mapping(A_sg, R_rg):
    cpu_M, fill_tree=get_eigen_matching(A_sg, R_rg);

    found=True
    #print repr(cpu_M)
    while is_acceptable(cpu_M) and found :
        P, row_ind, col_ind = get_hungarian_matching(cpu_M)
        mapping = compute_mapping(row_ind, col_ind, R_rg.shape[0])
        found, unfeasible, A_sg_upd = is_mapping_feasible(mapping, A_sg, R_rg, fill_tree)

        if found :
            cpu_M[unfeasible[0][0],unfeasible[0][1] ] = NO_MAPPING_VALUE

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
    return mapping, fill_tree, A_sg_upd

def is_mapping_feasible(mapping, A_sg, R_rg, fill_tree):
    A_sg_new = np.matrix(A_sg)
    unfeasible_mapping=[]
    for node_map in mapping :
        req_node=node_map[0]
        subs_node=node_map[1]
        #Map req node to substrate node and update CPU capacity
        A_sg_new[subs_node, subs_node] -= R_rg[req_node,req_node]

        #Determine all edges that request node connects and update their bandwidth capacities
        conn_nodes = list(np.argwhere((R_rg!= NO_LINK_VALUE)[req_node])[:,1])
        conn_nodes.remove(req_node) #Remove its own connection
        for node in conn_nodes:
            mapped_conn_node= mapping[node][1]
            if A_sg_new[subs_node,mapped_conn_node] == NO_LINK_VALUE:
                path = list(nx.shortest_simple_paths(fill_tree, subs_node, mapped_conn_node))[0]
                prev=subs_node
                for i in range(len(path) - 1):
                    A_sg_new[path[i], path[i+1]] -= R_rg[req_node, node]

            else:
                A_sg_new[subs_node, mapped_conn_node] -= R_rg[req_node, node]

            if np.any(A_sg_new<NO_LINK_VALUE) :
                unfeasible_mapping.append((node_map))
                return np.any(A_sg_new<NO_LINK_VALUE), unfeasible_mapping, A_sg

    return np.any(A_sg_new<NO_LINK_VALUE),unfeasible_mapping, A_sg_new

def is_acceptable(M) :
    mask = np.all(np.isinf(M), axis=1)
    if np.any(mask) :
        raise UnacceptableRequest( "Request cannot be satisfied as the following request nodes cannot be mapped: ",
                                   list(np.argwhere(mask)[:,0]+1))

    return  True


def fill_substrate_matrix(A_sg):

    G = nx.from_numpy_matrix(A_sg)

    T = nx_loc.maximum_spanning_tree(G)
    #print sorted(T.edges(data=True))

    #non_edges = np.argwhere(np.isnan(A_sg))
    non_edges = np.argwhere(A_sg==NO_LINK_VALUE)
    for ne in non_edges:
        p = list(nx.shortest_simple_paths(T, ne[0], ne[1]))[0]
        min_val = MAX_VALUE
        for i in range(len(p) - 1):
            temp = T.get_edge_data(p[i], p[i + 1])['weight']
            if temp < min_val:
                min_val=temp
        A_sg[ne[0],ne[1]] = min_val

    return A_sg, T

def get_test_graph_matrices(dir_name):
    return get_graph_matrices(dir_name, cs.SG_FILE_NAME), get_graph_matrices(dir_name, cs.RG_FILE_NAME)


def get_graph_matrices(dir_name, file_name):
    temp_graph = np.matrix(np.genfromtxt(cs.DATA_PATH+dir_name+'/'+file_name, delimiter=',', dtype=float))

    # nan means no edges between the nodes, we replace it by 0
    temp_graph[np.isnan(temp_graph)]=NO_LINK_VALUE

    return temp_graph



def compute_mapping(row_ind, col_ind, num_req_nodes):
    rg_nodes = map(list, zip(*[iter(row_ind)] * 1))
    sg_nodes = map(list, zip(*[iter(col_ind)] * 1))
    # print rg_nodes
    # print sg_nodes
    # print P
    mapping = np.hstack((rg_nodes, sg_nodes))
    req_mapping=mapping[:num_req_nodes]
    #print req_mapping

    return req_mapping

def get_mapping(A_sg, R_rg):
    try:
        mapping, fill_tree, A_sg=get_my_VRE_mapping(A_sg, R_rg)
        #print repr(mapping)
        #print repr(A_sg)
        #print repr(R_rg)
        #is_mapping_feasible(mapping, A_sg, R_rg, fill_tree)
        return  mapping, fill_tree, A_sg
    except Exception as error:
        #print('Caught this error: ' + repr(error))
        raise error

def persist_error(dir_name, file_name, error_status):
    with open(cs.DATA_PATH+dir_name+"/result_"+dir_name+".txt", "a") as myfile:
        myfile.write("\n"+file_name + "\tError\t" +error_status+"\t ")


def persist_result(dir_name, file_name, mapping, fill_tree):
    with open(cs.DATA_PATH+dir_name+"/result_"+dir_name+".txt", "a") as myfile:
        mapping_file_name="mapping_"+file_name+".txt"
        fill_tree_file_name="tree_"+file_name+".gexf"

        myfile.write("\n"+file_name + "\tSuccess\t" + mapping_file_name + "\t" + fill_tree_file_name)
        nx.write_gexf(fill_tree, cs.DATA_PATH+dir_name+"/"+fill_tree_file_name)
        np.savetxt(cs.DATA_PATH+dir_name+"/"+mapping_file_name, mapping)




def get_mapping_multiple_requests(dir_name) :
    import os
    #Load substrate graph
    A_sg_orig= get_graph_matrices(dir_name, cs.SG_FILE_NAME)
    A_sg = np.matrix(A_sg_orig)
    #Go through each of the requests graphs
    for subdir, dirs, files in os.walk(cs.DATA_PATH+dir_name):
        for file_name in files:
            if file_name.startswith(cs.RG_PREFIX) :
                try:
                    R_rg=get_graph_matrices(dir_name, file_name)
                    mapping, fill_tree, A_sg = get_mapping(A_sg=A_sg, R_rg=R_rg)
                    persist_result(dir_name, file_name, mapping, fill_tree)
                except Exception as error:
                    persist_error(dir_name, file_name, repr(error))
                    print('Caught this error: ' + repr(error))
                    pass




def test_mapping() :
    A_sg = np.matrix('3 7 0 3 1 0; '
                     '7 8 9 2 0 0 ;'
                     '0 9 15 4 10 0 ; '
                     '3 2 4 5 3 0; '
                     '1 0 10 3 11 5 ;  '
                     '0 0 0 0 5 2')

    #A_sg=fill_substrate_matrix(A_sg_init)

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

    print get_my_VRE_mapping(A_sg, R_rg)
    #print mapping
    #print is_mapping_feasible(mapping, A_sg, R_rg, T)

def test_matlab():
    A_sg = np.matrix('3 7 0 3 1 0; '
                     '7 8 9 2 0 0 ;'
                     '0 9 15 4 10 0 ; '
                     '3 2 4 5 3 0; '
                     '1 0 10 3 11 5 ;  '
                     '0 0 0 0 5 2')
    R_rg = np.matrix(' 8 5 2 ;5 1 0; 2 0 3 ')

    import matlab.engine
    eng = matlab.engine.start_matlab()
    cpu_M, fill_tree = get_eigen_matching(A_sg=A_sg, R_rg=R_rg)
    t=eng.mytest(cpu_M)



if  __name__ =='__main__' :
    #test_mapping()
    #A_sg, R_rg=get_test_graph_matrices(dir_name='rg_20_sg_15')
    #get_mapping(A_sg, R_rg)

    get_mapping_multiple_requests(dir_name='sg_20_rg_15')



    #test_matlab()
