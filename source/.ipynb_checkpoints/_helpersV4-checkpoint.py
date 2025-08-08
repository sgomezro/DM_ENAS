import numpy as np
from numpy.random import SeedSequence, default_rng
import json
import shutil
import torch
import os
import time
    
def load_parameters(p_filename,verbose=False):
    """Loads hyperparameters from .json file
    Args:
      p_filename - (string) - file name of hyperparameter file
      verbose  - (bool)   - print contents of hyperparameter file to terminal

    Note: see p/hypkey.txt for detailed hyperparameter description
    """    
    #Selectig the base line parameters
    p_default = "p/caeV4.json"
    with open(p_default) as data_file: p = json.load(data_file)
    with open(p_filename) as data_file: update = json.load(data_file)
    p.update(update)
    
    if verbose:
        print('\t*** Running with hyperparameters: ', pFileName, '\t***')

    return p
    
def lsave(filename, data):
    """Short hand for numpy save with csv and float precision defaults
    """
    np.savetxt(filename, data, delimiter=',',fmt='%1.2e')
    


def timeDisp(t):
    tarr = np.empty(3, dtype=int)
    tarr[2] = t
    for i in range(2):
        tarr[1-i] = int(tarr[2-i]/60)
        tarr[2-i] = tarr[2-i]%60

    return '{:4.0f}:{:02.0f}:{:02.0f}'.format(tarr[0],tarr[1],tarr[2])

def logSave(sys,log_save='null',stdout=True):
    #opening and saving log
    if not 'null' in log_save:
        logFile = open(log_save+'.txt', 'a')
        print('saving at:'+log_save+'.txt')
        stdout = sys.stdout
        sys.stdout = logFile
        return stdout
    else:
        return True
    #closing log and restoring stdout
    if not stdout:
        sys.stdout = stdout
        logFile.close()

# -- File I/O ------------------------------------------------------------ -- #
def export_agent(filename,ind):
    ''' exports agents data to recreate individual'''
    agent_data = np.array([np.array(ind.conn),np.array(ind.node)],dtype=object)
    np.save(filename, agent_data, allow_pickle=True)
    
# def importInd(filename,rng,w_fname=None,verbose=True):
#     ''' imports data to gererate agent individual data'''
#     from .individuals import Individuals
#     ind  = np.load(filename, allow_pickle=True)
#     conn = ind[0]
#     node = ind[1]
#     gen  = ind[2]
    
#     if w_fname != None:
#         weights = importWeights(w_fname)
#         ind = Individuals(conn,node,rng,gen,weights)
#         if verbose:
#             print('using weights from: '+w_fname)
#     else:
#         ind = Individuals(conn,node,rng,gen)
#     ind.express()
    
#     return ind

def exportNet(filename,weights,vKey,aVec,node_track):
    data = np.array([weights,vKey,aVec,node_track],dtype=object)
    np.save(filename+'.npy', data, allow_pickle=True)
    
def importNet(filename):
    mat_data = np.load(filename+'.npy', allow_pickle=True)
    weights = mat_data[0]
    vKey = mat_data[1]
    aVec = mat_data[2]
    node_track = mat_data[3]
    num_nod = len(aVec)

    wVec = np.zeros((num_nod**2))
    wVec[vKey] = weights
    
    return wVec,aVec,vKey,node_track

def save_adjw_files(fname,adjw_fname,weights,agent):
    '''to save adjusted weights on files 
    '''
    #Saving NN matrix for best weights adjustment
    exportWeights(adjw_fname,weights)
    # exportNet(fname,weights,agent.keys,agent.activations,agent.node_track)

#----- check architectures similarity--------
# def is_new_architecture(p,prev_p):
#     # Check if current nn architecture is equal to previous generation architecture
#     current_arch_file = p['experiment_path']+p['filename']
#     previous_arch_file = prev_p['experiment_path']+prev_p['filename']

#     if are_same_architectures(current_arch_file,previous_arch_file) &\
#         (prev_p['best_folder'] == p['best_folder']) &\
#         (prev_p['cma_init_best_sw'] == p['cma_init_best_sw']):
#         print('-> Copy CMA --Current NN with same structure as previous generation')
#         source_cma_f = prev_p['best_folder']+prev_p['cma_weights']+'.npy'
#         dest_cma_f = p['best_folder']+p['cma_weights']+'.npy'
#         shutil.copy(source_cma_f,dest_cma_f)
#         return False
#     return True

def is_new_gen_architecture(p,current_g,previous_g):
    # Check if current nn architecture is equal to previous generation architecture
    current_arch_file  = '{}{}.npy'.format(p['arch_fname'],current_g)
    previous_arch_file = '{}{}.npy'.format(p['arch_fname'],previous_g)

    if are_same_architectures(current_arch_file,previous_arch_file):
        return False
    return True

def are_same_architectures(file_a,file_b):
    '''compare two nn files and return if they have the same estructure and nodes,
        independent of the generation of each nn file'''
    result = False
    if (file_a != None) & (file_b != None):
        a = np.load(file_a, allow_pickle=True)
        b = np.load(file_b, allow_pickle=True)
        if (a[0].shape == b[0].shape) & (a[1].shape == b[1].shape):
            result = ((a[0] == b[0]).all()) & ((a[1] == b[1]).all())

    return result
    
def export_weights(filename,weights):
    np.save(filename+'.npy', weights, allow_pickle=True)
    
def importWeights(filename):
    return np.load(filename+'.npy', allow_pickle=True)


# def generateIndexJobs(n,size):
#     ''' Creates a index to subdivide population into vectors of subpopulation
#     '''
#     if size%n == 0:
#         nJobs = int(size/n)
#     else:
#         nJobs = int(size/n)+1
#     index = [i for i in range(0, size, nJobs)]+[size]
#     return index

##------------ split datasets sections--------------

def split_jobs(lst, n):
    """
    Split a list in n parts to submit n jobs
    
    :param lst: List of elements to aplit into jobs
    :param n: number of jobs
    :return: list with jobs
    """
    division_len = len(lst) // n
    at_least = len(lst) % n
    offset = 0
    result = []
    
    for i in range(n):
        start_index = division_len * i + offset
        if i < at_least:
            end_index = start_index + division_len + 1
            offset += 1
        else:
            end_index = start_index + division_len
        result.append(lst[start_index:end_index])
        
    return [0]+result

def split_input_target_sets(inputs, targets, n_workers, temp_dir_path):
    ''' Divide input and target sets in equal parts to the number of workers,
        the sets are divided with stratefied split so all the classes have same
        number of true values. Sets for each worker are saved as files in a temporal
        folder, so each worker can load it later.
        
    '''
    # Ensure inputs and targets have the same number of rows
    assert inputs.shape[0] == targets.shape[0], "Inputs and targets must have the same number of rows"

    # Create the specified directory if it does not exist
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)

    file_list = [0]

    # Calculate the size of each chunk
    chunk_size = inputs.shape[0] // n_workers

    time_id = int(time.time() * 1000)
    workers_indices = stratified_split(targets,n_workers)
    for i,idx in enumerate(workers_indices):
        # Slice the arrays
        input_chunk = inputs[idx]
        target_chunk = targets[idx]

        # Save the chunks to files
        input_filename = f'{temp_dir_path}{time_id}_worker_{i}_input.npy'
        target_filename = f'{temp_dir_path}{time_id}_worker_{i}_target.npy'
        np.save(input_filename, input_chunk)
        np.save(target_filename, target_chunk)

        # Add the filenames to the list
        file_list.append((input_filename, target_filename))

    return file_list

def stratified_split(targets, num_workers):
    """ Ensures all classes have the same number of true values, to avoid miss calculations
        when loss functions are calculated.
    """
    # Number of samples per worker
    samples_per_worker = int(torch.ceil(torch.tensor(targets.shape[0] / num_workers)).item())
    
    # Split indices for each worker
    worker_indices = [[] for _ in range(num_workers)]

    # Sort classes by the number of samples in ascending order
    class_counts = torch.sum(targets, axis=0)
    sorted_classes = torch.argsort(class_counts)

    # Distribute samples of each class to workers
    for class_idx in sorted_classes:
        class_indices = torch.where(targets[:, class_idx] == 1)[0]

        # Distribute indices among workers
        for i, idx in enumerate(class_indices):
            worker_indices[i % num_workers].append(idx.item())

    # Combine indices for each worker and trim excess
    for i in range(num_workers):
        worker_indices[i].sort()
        worker_indices[i] = worker_indices[i][:samples_per_worker]

    return worker_indices

def remove_tmp_worker_sets(fnames_l):
    ''' Remove temporal input and target files
    '''
    for tup_fnames in fnames_l[1:]:
        os.remove(tup_fnames[0])
        os.remove(tup_fnames[1])

#-----------others-------------------------

def get_device_by_rank(n_workers:int,gpus:torch.Tensor,rank:int,verbose:bool=False):
    ''' Confirms if there are cudas and have been enabled to compute
        with cudas during the fitness calculation. Also, defining 
        which will be the cuda Id.
        args:
        nWorkers = number or Cpu workers assigned to the task
        enableGpu= boolean flag
        
     '''
    device = torch.device('cpu')
    if rank > 0 and torch.cuda.is_available():
        # n_workers,gpus = p['n_workers'],p['gpus']
        gpu_id = -1
        #checkig available vs requested gpus
        gpus_on_server = [i for i in range(torch.cuda.device_count())]
        gpus_on_server = torch.tensor(gpus_on_server,dtype=torch.int32)
        gpus_to_use =  np.intersect1d(gpus,gpus_on_server)

        n_gpus = len(gpus_to_use)
        #getting the gpu ID
        if n_gpus > 0:
            gpu_id = [gpus_to_use[int(i*n_gpus/n_workers)] for i in range(n_workers)][rank]
            device = torch.device('cuda:'+str(gpu_id))
    if verbose:
        print('Worker: {} working with: {} for calculation'.format(rank,device))
    return device

def generate_rngs(seed,n_workers):
    # creating the RNG to pass around
    ss = SeedSequence(seed)

    rng_master,rng_workers = [],[]
    # Spawn off nWorkers child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(n_workers)
    rng_workers = [default_rng(s) for s in child_seeds]
    rng_master = rng_workers[0]
    if n_workers > 0:
        return rng_master,rng_workers
    else:
        return rng_master
    



