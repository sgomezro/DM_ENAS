import torch
import numpy as np
from ._individualsV4 import Individuals
from ._helpersV4 import importWeights

class nn_agent():
    def __init__ (self,input_size,device,ind=None,dtype=None):
        self.input_size = input_size
        self.dtype = dtype
        # self.device = device
        self.keys = []
        self.nn_weights = []
        self.activations= []
        self.num_nodes  = []
        self.node_track = []
        self.nn_matrix  = []
        if ind != None:
            self.from_ind(ind)
        
    def from_ind(self,ind):
        self.keys = ind.vKey
        self.activations= np.array(ind.aVec,dtype=np.int32)
        self.num_nodes  = ind.nNodes
        self.node_track = ind.node_track

    def set_nn_vector(self):
        self.w_vec_temp = torch.zeros(self.num_nodes**2,\
                                            dtype=self.dtype)
    
    # def shared_weight(self,weights):
    #     self.w_vec_temp[self.keys] = torch.tensor(weights,\
    #                                         dtype=self.dtype)
    #     self.nn_matrix = torch.reshape(self.w_vec_temp,\
    #                                             (self.num_nodes,self.num_nodes))
    
    def import_individual(self,filename,w_fname=None,verbose=True):
        ''' imports data to gererate nn agent'''
        ind  = np.load(filename, allow_pickle=True)
        conn = ind[0]
        node = ind[1]
        # gen  = ind[2]

        if w_fname != None:
            weights = importWeights(w_fname)
            if weights.dtype != np.float64:
                if 'Unstable for CMA' in weights:
                    print('Generation is unstable for CMA Adjustment. Continuing with saved weigths')
                    ind = Individuals(conn,node,None)
                else:
                    raise Exception('Error with cma weights.')
            else:
                ind = Individuals(conn,node,None,weights=weights)
            
            if verbose:
                print('using weights from: '+w_fname)
        else:
            ind = Individuals(conn,node,None)
        if not(ind.express()):
            raise Exception('Error importing nn architecture.')
        else:
            self.nn_w_vector= torch.tensor(ind.wVec,\
                                           dtype=self.dtype)
            self.nn_matrix  = torch.tensor(ind.wMat,\
                                           dtype=self.dtype)
            self.activations = np.array(ind.aVec,dtype=np.int32)
            self.keys  = ind.vKey
            self.num_nodes   = ind.nNodes
            self.node_track  = ind.node_track
            del ind
        
    def set_adj_weights(self,agent_weights):
        w_vec_temp = torch.zeros(self.num_nodes**2,\
                                       dtype=self.dtype)
        w_vec_temp[self.keys] = torch.tensor(agent_weights, dtype=self.dtype)
        self.nn_matrix = torch.reshape(w_vec_temp,(self.num_nodes,self.num_nodes))
        
    