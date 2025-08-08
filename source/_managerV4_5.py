from ._helpersV4 import *
from .fitnessFunctionsV4_5 import fitness_functions
from .loaders.loaderV4_5 import loader
from .F.taskLossClassifierV4 import loss_classifier

from .caeV4_5 import cae_class
from .dataGathererV4 import gatherer_class

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import os, subprocess, sys
import time
import numpy as np
from torch import double,cuda
import torch
import cma
import pickle
import shutil
from .timer import Timer

def cae_master(list_p,dtype=double):
    t = Timer()
    t.start()

    #Creating temporal adjw folder to save datasets for each worker
    time_id = int(time.time() * 1000)
    tmp_folder = list_p[0]['experiment_root']+f'tmp/cae_{time_id}/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder) 
        
    comm.bcast(list_p[0],root=0)
    comm.bcast(tmp_folder,root=0)
    for i,p in enumerate(list_p):
        print('\nRunning experiment {}'.format(p['exp_print']))        
        
        step = 'new_dataset'
        step = comm.bcast(step,root=0)
        p = comm.bcast(p,root=0)
        dataset = loader(p,'trainset',dtype=dtype)
        
        #generating random generators for workers
        rng_master,rng_workers = generate_rngs(p['seed'],p['n_workers'])
        #initiating new experiment
        step = 'new_experiment'
        step = comm.bcast(step,root=0)
        p = comm.bcast(p,root=0)
        #sending spawned rngs to each worker
        comm.scatter(rng_workers, root=0)

        fn_warm_adjw = get_pre_adjw(p,rng_master,verbose=True)
        data = gatherer_class(p,fn_warm_adjw)
        alg  = cae_class(p,rng_master,dtype=dtype)

        #Evolving architecture process
        while True:
            children,parents= alg.ask_cae()
            gen = alg.iteration
            if parents != None:
                step = 'breed'
                step = comm.bcast(step,root=0)
                gen  = comm.bcast(gen,root=0)

                scat_children = split_jobs(parents,p['n_slaves'])
                comm.scatter(scat_children, root=0)
                
                # gather_children = comm.gather(0,root=0)
                gather_fnames = comm.gather(0,root=0)
    
                for i in range(1,p['n_workers']):
                    with (open(gather_fnames[i], "rb")) as openfile:
                        children_load = pickle.load(openfile)
                    children.extend(children_load)
                alg.set_population(children)
                scat_children = None
                gather_children = None

            nn_population = alg.get_nn_population()     
            start_idx = dataset.get_train_start_idx(rng_master)
            inputs,targets = dataset.trainset_from_start_idx(start_idx)

           #MPI broadcasting workers to calculate fitness
            step = 'fitness_train'
            step = comm.bcast(step,root=0)
            # start_idx = comm.bcast(start_idx, root=0)
            inputs = comm.bcast(inputs,root=0)
            targets= comm.bcast(targets,root=0)
            
            fitness = request_fitness(nn_population,p['n_slaves'],is_vector=False)
            alg.tell(fitness)           # Send fitness to cae
            data.gather_data(alg.pop,gen)
            cuda.empty_cache()
            data.display(gen)
            
            if gen % p['save_mod'] == 0:
                data.save(gen)
            
            if alg.stop():
                data.save(gen)
                # sending step subprocess to stop all workers
                step = 'end_experiment'
                step = comm.bcast(step, root=0) #stops all workers
                break
    #ending jobs, stopping workers
    shutil.rmtree(tmp_folder)
    step = 'stop_workers'
    step = comm.bcast(step, root=0) #stops all workers
    t.stop()
        


def cae_worker(dtype=double):
    p = comm.bcast(None,root=0)
    tmp_folder = comm.bcast(None,root=0)
    device = get_device_by_rank(p['n_workers'],p['gpus'],rank,verbose=True)
    while True:
        step = comm.bcast(None,root=0)
        if step == 'new_dataset':
            p = comm.bcast(None,root=0)
            # Setting up for adjw phase
            # dataset_adjw = loader(p,'adj_w',dtype=dtype)
            # inputs_adjw,targets_adjw = dataset_adjw.adj_weight_set()

        elif step == 'new_experiment':
            p = comm.bcast(None,root=0)
            #receiving rng from Master
            rng_worker = comm.scatter(None, root=0)
            fit_train = fitness_functions(p,rank,set_type='trainset')
            loss_f = loss_classifier(p,rank).loss_function()
            fit_train.set_loss_f(loss_f)

            #new experiment for adj weights phase
            fit_adjw = fitness_functions(p,rank,set_type='adjw')
            loss_adjw = loss_classifier(p,rank).loss_function()
            fit_adjw.set_loss_f(loss_adjw)
            # fit_adjw.set_dataset(inputs_adjw,targets_adjw)

        elif step == 'breed':
            sub_children = []
            gen = comm.bcast(None, root=0)
            sub_population = comm.scatter(None, root=0)
            for ind in sub_population:
                child = ind.createChild(p, rng_worker,gen)
                child.express()
                sub_children.append(child)
            fname = f'{tmp_folder}worker{rank}_sub_children.pkl'
            with open(fname,'wb') as file:
                pickle.dump(sub_children,file,pickle.HIGHEST_PROTOCOL)
            # comm.gather(sub_children,root=0)
            comm.gather(fname,root=0)

        elif step == 'fitness_train':
            sub_fitness = None
            # start_idx     = comm.bcast(None, root=0)
            inputs_train  = comm.bcast(None,root=0)
            targets_train = comm.bcast(None,root=0)
            sub_population= comm.scatter(None, root=0)

            fit_train.set_dataset(inputs_train,targets_train)
            if len(sub_population) > 0:
                sub_fitness = fit_train.cae_shared_w_fitness(sub_population)
                sub_fitness = sub_fitness.cpu().numpy()
            gather_fitness = comm.gather(sub_fitness,root=0)

        elif step == 'set_adjw_nn':
            filename = comm.bcast(None,root=0)
            x_tensor = comm.bcast(None,root=0)
            y_tensor = comm.bcast(None,root=0)
            fit_adjw.load_nn_architecture(filename)
            fit_adjw.set_tensors(x_tensor,y_tensor)
            fit_adjw.ready_to_calculate()
            
        elif step == 'fitness_adjw':
            sub_weights = comm.scatter(None, root=0)
            sub_fitness = None
            if len(sub_weights) > 0:
                sub_fitness = fit_adjw.multiple_fitness(sub_weights)
                sub_fitness = sub_fitness.cpu().numpy()
            gather_fitness = comm.gather(sub_fitness,root=0)
            fit_adjw.set_loss_f(None)
            fit_adjw.set_loss_f(loss_adjw)

        elif step == 'end_experiment':
            sub_population = None
            rng_worker   = None
            inputs_train = None
            targets_train= None
            sub_fitness  = None
            fit_train = None
            fit_adjw  = None
            cuda.empty_cache()
        
        elif step == 'stop_workers':
            # del dataset
            break
            
def get_pre_adjw(p,rng,verbose=False,dtype=double):
    warm_adjw_maxiter = 200
    init_sigma = 1
    seed = p['seed']

    #Confirming sign of fitness to minimize or maximize CMA
    direction_sign = 1
    text = 'Minimizing'
    if p['opt_direction'] == 'maximize':
        text = 'Maximizing'
        direction_sign = -1
    print('{} warm adjw'.format(text))
    # fit_adjw = fitness_functions(p,rank)
    dataset_adjw = loader(p,'adj_w',dtype=dtype)
    inputs_adjw,targets_adjw = dataset_adjw.adj_weight_set()
    fit_adjw = fitness_functions(p,rank,set_type='adjw')
    fit_adjw.set_dataset(inputs_adjw,targets_adjw)
    
    def get_pre_adjw_fitness():
        filename = '{}{}/tmp.npy'.format(p['experiment_path'],p['filename'])
        fit_adjw.load_nn_architecture(filename)
        x_tensor, y_tensor = fit_adjw.get_tensors()
        step = 'set_adjw_nn'
        step = comm.bcast(step,root=0)
        comm.bcast(filename,root=0)
        comm.bcast(x_tensor,root=0)
        comm.bcast(y_tensor,root=0)
        
        #Getting best shared weights value for initialization
        step = 'fitness_adjw'
        step = comm.bcast(step,root=0)
        shared_weights = np.repeat(np.expand_dims(fit_adjw.wValList,axis=1),len(fit_adjw.agent.keys),axis=1)
        sw_fitness = request_fitness(shared_weights,p['n_slaves'],sign=direction_sign)
        best_sw_idx = sw_fitness.argmin()

        # print(f'best sw {best_sw_idx},\n sw fitness = {sw_fitness}')
        
        #creating inital weights with best shared weight value
        ibsw_w = np.ones(len(fit_adjw.agent.keys))*fit_adjw.wValList[best_sw_idx]
        initial_weights = rng.normal(ibsw_w, abs(0.1*ibsw_w))  # adding gausian noise

        if len(initial_weights) <= 1:
            print('Initial weights is an scalar, skipping CMA-ES because is unstable.')
            reward = sw_fitness[best_sw_idx]
            weights= ibsw_w

        else:
            cma_config = {'maxiter': warm_adjw_maxiter,'seed':seed,'verbose':1}#-9}
            es = cma.CMAEvolutionStrategy(initial_weights,init_sigma, cma_config)
            while not es.stop():
                step = 'fitness_adjw'
                step = comm.bcast(step,root=0)
    
                adj_weights = es.ask()
                fitness = request_fitness(adj_weights,p['n_slaves'],sign=direction_sign)
                es.tell(adj_weights,fitness)
                if verbose:
                    es.disp()
    
            reward = es.best.f*direction_sign
            weights = es.best.x
        return reward,weights
        
    return get_pre_adjw_fitness

            
def cma_master(list_p,dtype=double):
    p = comm.bcast(list_p[0],root=0)
    load_t  = Timer()
    cma_t   = Timer()
    total_t = Timer()
    total_t.start()
    
    #Creating temporal adjw folder to save datasets for each worker
    tmp_folder = list_p[0]['experiment_path']+'tmp/adjw/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder) 
        
    for init,p in enumerate(list_p):
        load_t.start()
        
        if p['is_new_dataset']:
            dataset = None
            dataset = loader(p,'adj_w',dtype=dtype)
            is_run_experiment = True

        if p['is_new_inputs']:
            if init > 0:
                target_adjw = None
                inputs_adjw = None
                remove_tmp_worker_sets(scat_fnames)
            dataset.update_window_size(p['window_size'])
            inputs_adjw,target_adjw = dataset.adj_weight_set()
            scat_fnames = split_input_target_sets(inputs_adjw,target_adjw,p['n_slaves'],tmp_folder)
            step = 'new_inputs'
            step = comm.bcast(step,root=0)
            comm.bcast(p,root=0)
            comm.scatter(scat_fnames, root=0)
            is_run_experiment = True

            #cleaning
            inputs_adjw = None
            target_adjw = None
            cuda.empty_cache()

        l_gen = p['gen_list']
        load_time = load_t.stop(verbose=False)
        for i,g in enumerate(l_gen):
            
            print('\nAdjusting weights for experiment {} gen {}'.format(p['exp_print'],g))
            if i == 0: #Initial setup
                #Confirming sign of fitness to minimize or maximize CMA
                direction_sign = 1
                text = 'Minimizing'
                if p['opt_direction'] == 'maximize':
                    text = 'Maximizing'
                    direction_sign = -1
                    print(f'{text} cma')
                fit_adjw = fitness_functions(p,rank,set_type='adjw')
                is_run_experiment = True
            else:
                # Check if current nn architecture is equal to previous generation architecture
                is_run_experiment = is_new_gen_architecture(p,g,l_gen[i-1])

            if not is_run_experiment:
                print(f'-> Copy CMA for {g}--Current NN with same structure as previous generation')
                #needs correction
                src_fname = '{}{}.npy'.format(p['adjw_fname'],l_gen[i-1])
                dst_fname = '{}{}.npy'.format(p['adjw_fname'],g)
                shutil.copy(src_fname,dst_fname)
            else:
                cma_t.start()
                arch_fname = p['arch_fname'] + g + '.npy'
                adjw_fname = p['adjw_fname'] + g 
                step = 'set_adjw_nn'
                comm.bcast(step,root=0)
                comm.bcast(arch_fname,root=0)
                fit_adjw.load_nn_architecture(arch_fname)
                
                rng,_ = generate_rngs(p['seed'],p['n_workers'])
                if not p['cma_init_best_sw']: # random initial guess
                    initial_weights = (rng.random(len(fit_adjw.agent.keys))*2-1)*p['cae_Sw_lim']
                else:
                    #Getting best shared weights value for initialization
                    step = 'fitness'
                    comm.bcast(step,root=0)
                    shared_weights = np.repeat(np.expand_dims(fit_adjw.wValList,axis=1),len(fit_adjw.agent.keys),axis=1)
                    sw_fitness = request_fitness(shared_weights,p['n_slaves'],sign=direction_sign)
                    best_sw_idx = sw_fitness.argmin()
                    #creating inital weights with best shared weight value
                    ibsw_w = np.ones(len(fit_adjw.agent.keys))*fit_adjw.wValList[best_sw_idx]
                    initial_weights = rng.normal(ibsw_w, abs(0.1*ibsw_w))  # adding gausian noise

                if len(initial_weights) <= 1: 
                    # initial weights lenght is unstable for cma adjustment
                    weights = 'Unstable for CMA'
                    export_weights(adjw_fname,weights)
                
                else:
                    # Initial weights are enought for cma adjustment
                    cma_config = {'popsize': p['popSize'], 'maxiter': p['cma_maxGen'],'seed':p['seed']}
                    es = cma.CMAEvolutionStrategy(initial_weights, p['cma_initSigma'], cma_config)
                    
                while not es.stop():
                    step = 'fitness'
                    comm.bcast(step,root=0)
        
                    adj_weights = es.ask()
                    fitness = request_fitness(adj_weights,p['n_slaves'],sign=direction_sign)
                    es.tell(adj_weights,fitness)
                    es.disp()

                    if es.result.iterations%p['save_mod_adjw'] == 0:
                        weights = es.best.x
                        export_weights(adjw_fname,weights)

                    #cleaning
                    fitness = None
                    adj_weights = None
                    cuda.empty_cache()
                    
                adjw_t = cma_t.stop(verbose=False)
                weights = es.best.x
                export_weights(adjw_fname,weights)
                
                print('For exp {} gen {} load time {:0.4f}, adjw time {:0.4f}'.format(p['exp_name'],g,load_time,adjw_t))
                
    #removing workers tmp files
    remove_tmp_worker_sets(scat_fnames)
    
    #Stopping workers
    step = 'stop_worker'
    comm.bcast(step, root=0)
    print(f'Total elapsed time {total_t.stop(verbose=False)}')
    load_t = None
    cma_t = None
    total_t = None


def cma_worker(dtype=double):
    p = comm.bcast(None,root=0)
    get_device_by_rank(p['n_workers'],p['gpus'],rank,verbose=True)

    while True:
        step = comm.bcast(None,root=0)
        if step == 'new_inputs':
            p = comm.bcast(None,root=0)
            inputs_fname,target_fname = comm.scatter(None, root=0)

            #cleaning
            inputs  = None
            target  = None
            fit_adjw= None

            # Loading the data
            inputs = torch.from_numpy(np.load(inputs_fname))
            target = torch.from_numpy(np.load(target_fname))

            #generating the fitness function
            fit_adjw = fitness_functions(p,rank,set_type='adjw')
            loss_adjw = loss_classifier(p,rank).loss_function()
            fit_adjw.set_loss_f(loss_adjw)
            fit_adjw.set_dataset(inputs,target)


        elif step == 'set_adjw_nn':
            arch_fname = comm.bcast(None,root=0)
            fit_adjw.load_nn_architecture(arch_fname)
        
    
        elif step == 'fitness':
            sub_weights = comm.scatter(None, root=0)
            sub_fitness = fit_adjw.multiple_fitness(sub_weights)
            gather_fitness = comm.gather(sub_fitness.cpu().numpy(),root=0)

            #cleaning
            sub_weights = None
            sub_fitness = None
            fit_adjw.set_loss_f(None)
            fit_adjw.set_loss_f(loss_adjw)
            
        elif step == 'stop_worker':
            fit_adjw = None
            break

def request_fitness(weights_v,n_slaves, sign=1, is_vector=True):
    '''
        Calculate fitness for the different n jobs weights,
        sending the request the portion of the job to each worker.
        param weights_v: list with weights for each job
        param n_slaves : number of workers to split the jobs
        param sign: optimization direction for the fitness
        param is_vector: flag to define the way stack the results
        return: numpy array of fitness results
        
    '''
    v_size = len(weights_v)
    gather_end = n_slaves+1 if v_size > n_slaves else v_size+1
    scatter = split_jobs(weights_v,n_slaves)
    comm.scatter(scatter, root=0)
    gather = comm.gather(0,root=0)

    if is_vector:
        fitness = np.hstack(gather[1:gather_end])
        
    else:
        fitness = np.vstack(gather[1:gather_end])
    return fitness*sign

def cmaSlave_predict(error_fun=None):
    p0 = comm.bcast(None,root=0)
    dataset = loader(p0,'cma')
    fit = fitness_functions(p0,rank)
    if error_fun == None:
        fit.setFunError(fit.calculateError)
    else:
        exec('fit.setFunError('+error_fun+')')

    print('Worker: ',rank,' ',fit.workingWith)

    
    while True:
        step = comm.bcast(None,root=0)
        if step == 'new_generation':
            p = comm.bcast(None,root=0)
            
            sub_population = None
            cmaInput,cmaTarget = dataset.getCmaSet()
    
            #setting partial output CAN BE IMPROVED
            fit.set_dataset(cmaInput,cmaTarget)
            fit.loadNNMat(p['experiment_path']+p['filename'])
    
        elif step == 'fitness':
            sub_population = comm.scatter(sub_population, root=0)
            sub_fitness = np.zeros((len(sub_population)))

            for i,agentW in enumerate(sub_population):
                sub_fitness[i] = fit.getCmaAgentFitness(agentW)

            gather_fitness = comm.gather(sub_fitness,root=0)

        elif step == 'stop_worker':
            del sub_population
            del sub_fitness
            del fit
            del dataset
            del cmaInput
            del cmaTarget
            break
            
            
def mpi_fork(n_cpus):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    [modified from https://github.com/garymcintire/mpi_util.git] 
    """
    end_proc = False
    if n_cpus<=1:
        return end_proc

    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
          MKL_NUM_THREADS="1",
          OMP_NUM_THREADS="1",
          IN_MPI="1"
          )
        subprocess.check_call(["mpiexec", "-n", str(n_cpus), sys.executable] +['-u']+ sys.argv, env=env)
        end_proc = True
    return end_proc


    