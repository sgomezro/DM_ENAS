import numpy as np
import torch
import torch.nn.functional as F
import os
import subprocess
import time
from ._agentV4 import nn_agent 
from ._helpersV4 import get_device_by_rank, split_jobs

DEBUG = False

class fitness_functions():
    def __init__ (self,p,rank,verbose=False,dtype=torch.double,set_type=None):
        self.input_size  = p['nn_input_size']*p['window_size']
        self.model_type = p['ENAS_type']
        self.rank = rank
        self.batch_size = p['batch_size']


        # Settting share weight values to test the fitness of the architecture
        swLim   = p['cae_Sw_lim']
        nVals   = p['cae_Sw_nVals']
        wValList = np.linspace(-swLim, swLim ,nVals+1,dtype=np.double)
        self.wValList = wValList[wValList !=0]

        # Setting cpu or gpu device
        device = get_device_by_rank(p['n_workers'],p['gpus'],rank,verbose=verbose)
        self.device = device
        #Setting GPU configuration
        self.gpu_active = False
        if p['gpus']:
            self.init_gpu_mem = 0
            self.pid = os.getpid()
            self.gpu_active = True

        self.check_mem = gpu_memory_fun(p['gpus'],p['gpu_mem'],p['n_slaves'],p['batch_size'])

        self.set_type = set_type        
        self.dtype  = dtype
        
        # Setting output filter
        if 'output_filter' in p:
            self.filter_type = p['output_filter']
        else:
            self.filter_type = None

        self.y = []
        self.x = []

        
        
   
    def set_loss_f(self,loss_f):
        self.loss_fun = loss_f
        
    def set_dataset(self,x,y):
        # print(f'memory in evaluate agent x {len(x)}')
        self.x = x
        self.y = y.to(self.device)

    def get_tensors(self):
        return self.tensor_x, self.y

    def set_tensors(self,x_tensor,y_tensor):
        self.tensor_x = None
        self.y = None
        torch.cuda.empty_cache()

        self.tensor_x = x_tensor.to(self.device)
        self.y = y_tensor.to(self.device)
        self.get_init_gpu_mem()


    def ready_to_calculate(self):
        ''' Check requiriments before proceeding with fitness calculation
        '''
        flag = False
        if self.loss_fun == None:
            raise Exception('loss_fun is not defined, please define it using set_loss_f.')
        elif not((self.model_type == 'aDetection') | (self.model_type == 'prediction')):
            raise Exception('ENAS type should be aDetection or prediction but you gave {}'.\
                            format(self.model_type))
        else:
            flag = True
        return flag

       
    def load_nn_architecture(self, filename,w_fname=None,verbose=True):
        ''' Load nn architecture specifications as an agento into the fitness class
        '''
        self.agent = nn_agent(self.input_size,self.device,dtype=self.dtype)
        self.agent.import_individual(filename,w_fname=w_fname,verbose=verbose)

        if len(self.x) > 0:
            self.input_to_tensor(self.x,self.agent.num_nodes)
        


    def cae_shared_w_fitness(self,agents_list):
        '''Calculates neural architecture agent fitness for a list of agents
        '''
        agents_fitness = None
        procced_with_calculation = self.ready_to_calculate()
        if procced_with_calculation:
            num_agents = len(agents_list)
            agents_fitness = torch.zeros((num_agents,self.wValList.shape[0]),\
                                        dtype=self.dtype)
    
            for a, agent in enumerate(agents_list):
                self.agent = agent
                self.agent.set_nn_vector()
                self.input_to_tensor(self.x,agent.num_nodes)
                # agent.set_nn_vector(self.device)
                
                
                # Get reward from rollouts -- test population on same seed
                shared_weights = np.repeat(np.expand_dims(self.wValList,axis=1),\
                                           len(self.agent.keys),axis=1)
                agents_fitness[a,:] = self.multiple_fitness(shared_weights)#.cpu()
    
        return agents_fitness

    
    def adjw_agent_fitness(self,agent_weights):
        '''Calculates adjusted weights agent fitness
        '''
        self.agent.set_adj_weights(agent_weights)
        reward = None
        procced_with_calculation = self.ready_to_calculate()
        if procced_with_calculation:
            fitness = self.single_fitness()

        return fitness


    def evaluate_agent_fitness(self,filename,w_fname,verbose=False):
        ''' Calculates the evaluation agent performance.
            Used to calculate the metric evaluation.
        '''
        # self.load_nn_architecture(filename,w_fname=w_fname,verbose=verbose)
        self.agent = nn_agent(self.input_size,self.device,dtype=self.dtype)
        self.agent.import_individual(filename,w_fname=w_fname,verbose=verbose)

        x_size = len(self.x)
        split_size = self.check_mem(self,0,is_evaluating=True,verbose=verbose)

        predictions = torch.zeros_like(self.y)
        for i in range(0,x_size,split_size):
            sub_x = self.x[i:i + split_size]
            self.input_to_tensor(sub_x,self.agent.num_nodes)
            procced_with_calculation = self.ready_to_calculate()
            if procced_with_calculation:
                sub_y_hat = self.single_calculation(get_preds=True)

            predictions[i:i + split_size] = sub_y_hat#.cpu().numpy()
        
        return predictions

    
    def single_calculation(self,get_preds=False):
        ''' Calculate fitness for a single weight agent
        '''
        num_nodes  = self.agent.num_nodes
        input_size = self.agent.input_size
        nn_matrix  = self.agent.nn_matrix.to(self.device).to(self.dtype)
        activations= torch.from_numpy(self.agent.activations).to(self.device)
        node_track = self.agent.node_track
        filter = (node_track[2,:] == 2) 
        out_idx = node_track[0,filter]
        x = self.tensor_x.clone().to(self.device)

        nn_y = forward_nn_activation(num_nodes,input_size,nn_matrix,activations,x)

        output = nn_y[:,out_idx]
        y_hat =  multiple_filter_output(output,self.filter_type,dim=1)

        # Cleaning memory
        nn_matrix = None
        activations= None
        x = None
        nn_y = None
        output = None
        
        if get_preds:
            return y_hat
        else:
            fitness = self.loss_fun(self.y,y_hat).to(self.dtype)
            y_hat = None
            return fitness.unsqueeze(-1)
        

    
    def multple_calculation(self,agents_w_list,verbose=False):
        ''' Calculates fitness for multiples weights given the same 
            NN architecture.
        '''
        num_nodes  = self.agent.num_nodes
        input_size = self.agent.input_size
        # selecting indexes for outputs
        node_track = self.agent.node_track
        filter = (node_track[2,:] == 2) 
        out_idx = node_track[0,filter]

        num_agents = len(agents_w_list)
        sub_nn_mat = torch.empty([num_agents,num_nodes,num_nodes],\
                                  dtype=self.dtype,device=self.device)
        sub_activations = torch.empty((num_agents,num_nodes),\
                                       dtype=torch.int,device=self.device)
        sub_x = self.tensor_x.clone().repeat(num_agents,1,1).to(self.device)
        fitness = torch.zeros(num_agents,dtype=self.dtype,device=self.device)

        
        for i,weights in enumerate(agents_w_list):
            self.agent.set_adj_weights(weights)
            sub_nn_mat[i]  = self.agent.nn_matrix
            sub_activations[i]= torch.from_numpy(self.agent.activations)

        nn_y = multiple_forward_nn(num_nodes,input_size,sub_nn_mat,
                               sub_activations,sub_x)
        
        output = nn_y[:,:,out_idx]
        y_hat = multiple_filter_output(output,self.filter_type,dim=2)

        for i in range(num_agents):
            fitness[i] = self.loss_fun(self.y,y_hat[i])

        # Cleaning memory
        sub_nn_mat = None
        sub_activations= None
        sub_x = None
        nn_y = None
        output = None
        y_hat = None
        
        return fitness

    def multiple_fitness(self,all_agents_weights,verbose=False):
        '''Calculate multiple fitness for multiple agent
        '''
        #saving the NN structure to calculate fitness
        if DEBUG & (self.set_type == 'adjw') & (self.rank == 1):
            print(f'agent {len(self.agent.keys)}, weights {len(all_agents_weights[0])}')
        self.agent.set_adj_weights(all_agents_weights[0])
        reward = torch.tensor([],dtype=self.dtype,device=self.device)
        
        #Creating multiple matrices with length as adjw population
        num_all_agents = len(all_agents_weights)
        num_samples = len(self.x)
        
        # if self.set_type == 'adjw':
        #     n_split = self.check_mem(self,num_all_agents,verbose=verbose)
        # else:
        #     n_split = self.check_mem(self,num_all_agents)
        n_split = self.check_mem(self,num_all_agents)
        agent_weights_list = split_jobs(all_agents_weights,n_split)[1:]

        for sub_agents_w in agent_weights_list:
            if len(sub_agents_w) == 1: #calculate single fitness
                sub_reward = self.single_calculation()

            else: #Calculate multiple fitness
                sub_reward = self.multple_calculation(sub_agents_w,verbose=verbose)
                
                
            # reward = torch.cat((reward,sub_reward.cpu()),axis=0)
            reward = torch.cat((reward,sub_reward),axis=0)
            sub_reward = None
            torch.cuda.empty_cache()

        return reward

     
    def input_to_tensor(self,x,n_nodes):
        ''' Converts a dataset input into the tensor matrix with bias = 1
            for signal propagation through the NN matrix
            type torch tensor float
          Args:
            inData  - (torch.tensor) - Dataset input matrix [nSamples X inputnodes] 
            nNodes  - (int)   - Number of nuerons the NN matrix contains

          Returns:
            matrix_tensor - (torch tensor) - activation matrix with
                      [nSamples X nNodes] dimensionality 
        '''
        self.tensor_x = None
        torch.cuda.empty_cache()
        
        n_samples = x.shape[0]

        # locate input data through activation matrix    
        matrix_tensor  = torch.zeros((n_samples,n_nodes),dtype=self.dtype,device=self.device)
        matrix_tensor[:,0] = 1 # Bias activation
        matrix_tensor[:,1:self.input_size+1] = x

        self.get_init_gpu_mem()
        self.tensor_x = matrix_tensor
    
    def get_init_gpu_mem(self):
        if self.gpu_active:
            nvidia_smi_command = f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,nounits,noheader'
            processes_info = subprocess.check_output(nvidia_smi_command, shell=True).decode('utf-8').strip().split("\n")
            for process_info in processes_info:
                process_pid, memory_used = map(int, process_info.split(','))
                if process_pid == self.pid:
                    self.init_gpu_mem =  memory_used


def gpu_memory_fun(gpus_list,gpu_mem,n_slaves,b_size):
    """
    Defines if GPU is used or only CPU. 
    Get the GPU memory usage for a specific PID using the nvidia-smi tool.
    
    :param pid: Process ID to check.
    :return: Memory usage in MB for the given PID or None if PID is not using GPU.
    """
    if gpus_list:
        mem_worker = gpu_mem/n_slaves
        batch_size = b_size
        def check_gpu_memory(self,num_agents,is_evaluating=False,verbose=False):
            split_steps = 1
            nn_mat = self.agent.nn_matrix.to(self.device)
            act = torch.from_numpy(self.agent.activations).to(self.device)
            
            if is_evaluating:
                if (batch_size > 0) & (len(self.x) > batch_size):
                    self.input_to_tensor(self.x[:batch_size],self.agent.num_nodes)
                else:
                    self.input_to_tensor(self.x,self.agent.num_nodes)
            x = self.tensor_x.clone()
            yh = forward_nn_activation(self.agent.num_nodes,
                                      self.agent.input_size,
                                      nn_mat,act,x)

            # print(f'nn mat {nn_mat.dtype}, act {act.dtype}, x {x.dtype}, y {y.dtype}')
            mat_s = tensor_memory(nn_mat)
            act_s = tensor_memory(act)
            x_s   = tensor_memory(x)
            yh_s  = tensor_memory(yh)
            mem = self.init_gpu_mem+(mat_s+act_s+x_s+yh_s)*num_agents

            #cleaning tensors from memory
            nn_mat = None
            act = None
            x  = None
            yh = None

            if is_evaluating: # Determines the split size for an array 'x' so that it doesn't exceed the memory limit of a worker.
                # Calculate memory usage per row
                
                if batch_size > 0:
                    mem_row = (x_s + yh_s) / batch_size
                else:
                    mem_row = (x_s + yh_s) / len(self.x)

                # print(f'x s {x_s}, mem row {mem_row}')
                not_x_mem = self.init_gpu_mem+mat_s+act_s
                # Calculate maximum number of rows that can fit in mem_worker
                split_size = int((mem_worker - not_x_mem) / mem_row)
                if verbose & (self.rank == 1):
                    print(f'Worker {self.rank} with init mem of {self.init_gpu_mem}.')
                    print(f'GPU not x {not_x_mem}, memory row x {mem_row} mem per worker {mem_worker} split size {split_size}.')
                return split_size
                
            elif mem > mem_worker: # Calculating how many times fitness will be computed to fit in memory
                split_steps = int(np.ceil((mem-self.init_gpu_mem)/\
                                          (mem_worker-self.init_gpu_mem)))
                
                if verbose & (self.rank ==1):
                    print(f'Worker {self.rank} with init mem of {self.init_gpu_mem}.')
                    print(f'GPU memory required {mem} is greather than max allowed per worker {mem_worker} for {num_agents} agents.')
                    print(f'Number of fitness steps increased to {split_steps}.')

            return int(split_steps)


    else:
        def check_gpu_memory(self,num_agents,is_evaluating=False,verbose=False):
            return 1
                
    return check_gpu_memory





@torch.jit.script
def applyActivationTensor(actId:int, x):
    """Returns value after an activation function is applied
    Lookup table to allow activations to be stored in torch tensors

    case 1  -- Linear
    case 2  -- Unsigned Step Function
    case 3  -- Sin
    case 4  -- Gausian with mean 0 and sigma 1
    case 5  -- Hyperbolic Tangent [tanh] (signed)
    case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
    case 7  -- Inverse
    case 8  -- Absolute Value
    case 9  -- Relu


    Args:
        actId   - (int)   - key to look up table
        x       - (pytorch tensor)   - value to be input into activation
                  [? X ?] - any type or dimensionality

    Returns:
        output  - (float) - value after activation is applied
                  [? X ?] - same dimensionality as input
    """
    if actId == 1:   # Linear
        value = x

    elif actId == 2:   # Unsigned Step Function
        value = 1.0*(x>0.0)

    elif actId == 3: # Sin
        value = torch.sin(np.pi*x) 

    elif actId == 4: # Gaussian with mean 0 and sigma 1
        value = torch.exp(-(x*x) / 2.0)

    elif actId == 5: # Hyperbolic Tangent (signed)
        value = torch.tanh(x)     

    elif actId == 6: # Sigmoid (unsigned)
        value = (torch.tanh(x/2.0) + 1.0)/2.0

    elif actId == 7: # Inverse
        value = -x

    elif actId == 8: # Absolute Value
        value = torch.abs(x)   
    
    elif actId == 9: # Relu
        value = torch.maximum(torch.tensor(0), x)   

    else:
        raise ValueError ('must select a value between 1 and 9, {} is not allowed'.format(actId))

    return value


@torch.jit.script
def forward_nn_activation(num_nodes:int,input_size:int,nn_matrix,activations,ff_act):
    ''' Propagate signal through hidden to output nodes
        using nn agent information.
    '''
    for i_node in range(input_size+1,num_nodes):
        raw_act = torch.mv(ff_act, nn_matrix[:,i_node]).squeeze()
        activation_f = activations[i_node]
        ff_act[:,i_node] = applyActivationTensor(activations[i_node],raw_act)

    return ff_act

@torch.jit.script
def multiple_forward_nn(num_nodes:int,input_size:int,nn_matrix,activations,ff_act):
    ''' Propagate signal through hidden to output nodes
        using nn agent information.
    '''
    for i_node in range(input_size+1,num_nodes):
        raw_act = torch.matmul(ff_act, nn_matrix[:,:,i_node].unsqueeze(-1)).squeeze()
        for j in range(activations.shape[0]):
            act_id = activations[j,i_node]
            ff_act[j,:,i_node] = applyActivationTensor(act_id,raw_act[j])

    return ff_act

def multiple_filter_output(output,filter_type,dim):
    # print(f'filter type : {filter_type}')
    if (filter_type == None) | (filter_type == 'none') | (filter_type == 0):
        y_hat = output

    elif filter_type == 'mul_none':
        y_hat = F.normalize(output,dim=dim)
        
    elif (filter_type == 'softmax') | (filter_type == 1):
        y_hat = F.softmax(output,dim=dim) 
        
    elif (filter_type == 'mio') | (filter_type == 2):
        _, indices = output.max(dim=dim, keepdim=True)
        # Create a zero tensor of the same shape as the input tensor
        y_hat = torch.zeros_like(output)
        # Use scatter_ to set the maximum values in each 1x1xN slice to 1
        y_hat.scatter_(-1, indices, 1)
    else:
        raise Exception(f'Define a output filter from the following options: none, softmax or mio (max_is_one). Filter type: {filter_type} is provided.')

    # return y_hat.cpu()
    return y_hat


def tensor_memory(tensor):
    """
    Calculate the GPU memory usage for a given PyTorch tensor in megabytes.
    
    :param tensor: PyTorch tensor.
    :return: Memory usage in megabytes.
    """
    
    if not tensor.is_cuda:
        raise ValueError("The tensor is not on GPU.")
    
    # Calculate memory in bytes using tensor's properties
    memory_in_bytes = tensor.element_size() * tensor.nelement()
    
    # Convert to megabytes
    memory_in_MB = memory_in_bytes / (1024 * 1024)
    
    return memory_in_MB



