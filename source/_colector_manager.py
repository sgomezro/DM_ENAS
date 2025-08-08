import pandas as pd
import numpy as np
from source.fitness_functions import FitnessFunctions,calculateBynaryClassification,calculateMultiClassError
from source.load_anomaly_dataset import LoadADdataset
from source.metrics import Metrics
from ._helpers import *
from pathlib import Path

import os, subprocess, sys

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def colect_single_processor(l_p, exp_info,testing=False):
    for i,p in enumerate(l_p):
        if testing:
            p['data_path'] = '/storage_1/aDetection/data/s{}_20Hz_TEset_nonNorm_addedMissSqSat.csv'.format(p['sensor'])
            print('TESTING MODE: working with dataset {}'.format(p['data_path']))
            dataset = LoadADdataset(p,'testset')
            t_input,t_target = dataset.getTestSet()
            
        elif i == 0:
            path = Path(__file__).parents[4]
            print(path)
            p['data_path'] = '/storage_1/aDetection/data/s{}_20Hz_realset.csv'.format(p['sensor'])
            print('working with dataset {}'.format(p['data_path']))
            dataset = LoadADdataset(p,'testset')
            t_input,t_target = dataset.getTestSet()
        
        elif p['sensor'] != l_p[i-1]['sensor']:
            del dataset
            del t_input
            del t_target
            p['data_path'] = '/storage_1/aDetection/data/s{}_20Hz_realset.csv'.format(p['sensor'])
            print('changing to work with dataset {}'.format(p['data_path']))
            dataset = LoadADdataset(p,'testset')
            t_input,t_target = dataset.getTestSet()
        
        fit = FitnessFunctions(p)
        if p['classification_type'] == 'binary':
            fit.setFunError(calculateBynaryClassification)
        elif p['classification_type'] == 'multilabel':
            fit.setFunError(calculateMultiClassError)
        else:
            raise Exception('Only binary or multilabel classification types are allowed.')
        fit.setData(t_input,t_target)
        metrics = Metrics(p)
        
        exp    = exp_info[i][0]
        inputs = exp_info[i][1]
        nW     = exp_info[i][2]
        filter = exp_info[i][3]
        gen    = exp_info[i][4]
        exp_name=exp_info[i][5]
        
        print('calculating sensor {}, {}, gen {}, filter {}'.format(p['sensor'],exp_name,gen,filter))

        fit.evalInd(p['filename'],w_fname=p['cma_weights'])
        metrics.set_class_values(fit)
        result = metrics.get_results()
        if testing:
            c_name = 'test_s{}exp{}in{}nW{}g{}{}.csv'.format(p['sensor'],exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
        else:
            c_name = 'result_s{}exp{}in{}nW{}g{}{}.csv'.format(p['sensor'],exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
        if 'storage_path' in p:
            # appending storage 1 path to save directly at the server storage
            sys.path.append(p['storage_path'])
            save_filename = '{}{}/{}{}'.format(p['storage_path'],p['save_folder'],p['save_filename'],c_name)
            result.to_csv(save_filename)
        else:
            raise ValueError('It is neccesary to define the storage path')

def get_ECCD_scores(cl_sensor,l_sensor,end_dir,l_cat,n_rows=None):
    storage_path = '/storage_1/aDetection'
    p_adjust = 'experiments/aDetection_s{}_{}/aDetection.json'.format(cl_sensor,end_dir)
    p = loadParameters(p_adjust)

    l_exp  = [1,2,3,4,5]
    inputs = p['best_inputs']
    nW     = p['best_nW']
    filter = p['best_filter']
    gen    = str(p['best_gen']).zfill(4)
    p['nn_input_size'] = inputs
    p['target_size'] =  l_cat

    cat_scores = pd.DataFrame({})
    general_scores = pd.DataFrame({})
    for sensor in l_sensor:
        for exp in l_exp:
            col = ['centremost','sensor','exp','inputs','normal_weight','filter','gen']
            info = [cl_sensor,sensor,exp,inputs,nW,filter,gen]
            head = pd.DataFrame([info],columns=col)

            c_name = 'result_s{}exp{}in{}nW{}g{}{}.csv'.format(sensor,exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
            filename = '{}/experiments/s{}_{}/best_results/ECCD_s{}_{}'.format(storage_path,cl_sensor,end_dir,cl_sensor,c_name)
            metrics = Metrics(p)
            metrics.set_from_csv(filename,n_rows=n_rows)
            cat_rep = metrics.get_multi_score_report(score_get='full').T
            cat_rep.index.name='category'

            cat_report =pd.DataFrame({})
            columns = ['precision','recall','f1-score']
            cat_size = len(metrics.target_labels)
            aux = pd.merge(head.copy(),cat_rep.iloc[:cat_size,:3].reset_index(),how='cross')
            cat_report = pd.concat([cat_report,aux])
            cat_scores = pd.concat([cat_scores,cat_report])

            if end_dir == 'bin':
                gen_report = metrics.get_bin_score_report(score_get='one_row')
            elif (end_dir == 'filter') | (end_dir == 'weightned'):
                gen_report = metrics.get_multi_score_report(score_get='one_row')
            else:
                raise ValueError('It is neccesary to define end directory as bin, filter, or weightned')
            aux2 = pd.merge(head.copy(),gen_report,how='cross')
            general_scores = pd.concat([general_scores,aux2],axis=0)
            del metrics

    cat_scores.reset_index(inplace=True,drop=True)
    general_scores.reset_index(inplace=True,drop=True)

    path = 'experiments/aDetection_s{}_{}/score_results/'.format(cl_sensor,end_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    prefix = 'general_scores'
    filename = 's{}_cluster_{}.csv'.format(cl_sensor,prefix)
    general_scores.to_csv(path+filename)

    prefix = 'category_scores'
    filename = 's{}_cluster_{}.csv'.format(cl_sensor,prefix)
    cat_scores.to_csv(path+filename)
    
# deprecated    
# def get_send_dataset(p):
#     # Counting how many rows are in the csv file, and then divide in chunks to transmite to workers
#     dataset = LoadADdataset(p,'testset')
#     X,Y = dataset.getTestSet()
#     d_size = len(X)
#     chunk_size = (np.ceil(d_size/p['n_slaves'])).astype(long)
    
#     step = 'dataset'
#     step = comm.bcast(step,root=0)
    
#     l_idx = generateIndexJobs(p['n_slaves'],np.shape(X)[0])
#     scatter_X = [0]+[X[l_idx[i]:l_idx[i+1]] for i in range(p['n_slaves'])]
#     scatter_Y = [0]+[Y[l_idx[i]:l_idx[i+1]] for i in range(p['n_slaves'])]
#     comm.scatter(scatter_X, root=0)
#     comm.scatter(scatter_Y, root=0)
    
#     del dataset
#     del X
#     del Y
#     del scatter_X
#     del scatter_Y
#     return l_idx,d_size
    
# def master_colector(l_p, exp_info,testing=False):
#     p = comm.bcast(l_p[0],root=0)
# #     scores = pd.DataFrame({})
#     for i,p in enumerate(l_p):
#         if testing:

#             p['data_path'] = '/data/s{}_20Hz_TEset_nonNorm_addedMissSqSat.csv'.format(p['sensor'])
#             print('TESTING MODE: working with dataset {}'.format(p['data_path']))
#             l_idx,d_size = get_send_dataset(p)
#             os.chdir(orig_path)
            
#         elif i == 0:
#             p['data_path'] = '/data/s{}_20Hz_realset.csv'.format(p['sensor'])
#             print('working with dataset {}'.format(p['data_path']))
#             l_idx,d_size = get_send_dataset(p)
        
#         elif p['sensor'] != l_p[i-1]['sensor']:
#             p['data_path'] = '/data/s{}_20Hz_realset.csv'.format(p['sensor'])
#             print('changing to work with dataset {}'.format(p['data_path']))
#             l_idx,d_size = get_send_dataset(p)
                
#         step = 'parameters'
#         step = comm.bcast(step,root=0)
#         p = comm.bcast(p,root=0)
        
        
#         exp    = exp_info[i][0]
#         inputs = exp_info[i][1]
#         nW     = exp_info[i][2]
#         filter = exp_info[i][3]
#         gen    = exp_info[i][4]
#         exp_name=exp_info[i][5]
#         print('calculating sensor {}, {}, gen {}, filter {}'.format(p['sensor'],exp_name,gen,filter))
        
#         gather_results= comm.gather(0,root=0)
#         results =pd.DataFrame({})
#         for i in range(1,p['n_workers']):
#             results = pd.concat([results,gather_results[i]])
#         results.reset_index(drop=True,inplace=True)
        
#         if testing:
#             c_name = 'test_s{}exp{}in{}nW{}g{}{}.csv'.format(p['sensor'],exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
#         else:
#             c_name = 'result_s{}exp{}in{}nW{}g{}{}.csv'.format(p['sensor'],exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
#         data_path ='storage_1/aDetection/'
#         save_filename = '{}{}/{}{}'.format(p['storage_path'],p['save_folder'],p['save_filename'],c_name) 
#         results.to_csv(save_filename)
#     step = 'stop'
#     step = comm.bcast(step,root=0)
    

# def slave_colector():
#     p = comm.bcast(None,root=0)
#     gpuId = getGpuId(p['n_slaves'],p['n_gpus'],rank)
    
#     if p['classification_type'] == 'binary':
#         fun_class = calculateBynaryClassification
#     elif p['classification_type'] == 'multilabel':
#         fun_class = calculateMultiClassError
#     else:
#         raise Exception('Only binarry or multilabel classification types are allowed.')

#     X = None
#     Y = None
#     sub_X = None
#     sub_Y = None
#     while True:
#         step = comm.bcast(None,root=0)
#         if step == 'dataset':
#             sub_X = comm.scatter(sub_X, root=0)
#             sub_Y = comm.scatter(sub_Y, root=0)
#         if step == 'parameters':
#             p = comm.bcast(None,root=0) 
#             fit = FitnessFunctions(p,gpuId=gpuId)
#             fit.setFunError(fun_class)
#             fit.setData(sub_X,sub_Y)
#             metrics = Metrics(p)
#             fit.evalInd(p['filename'],w_fname=p['cma_weights'])
#             metrics.set_class_values(fit)
#             sub_results = metrics.get_results()
# #             results_label = comm.gather(sub_results.columns,root=0)
#             gather_results = comm.gather(sub_results,root=0)
#             del metrics
#             del fit
#             del sub_results
#         if step == 'stop':
#             break
        
        
# def mpi_fork(n_cpus):
#     """Re-launches the current script with workers
#     Returns "parent" for original parent, "child" for MPI children
#     [modified from https://github.com/garymcintire/mpi_util.git] 
#     """
#     end_proc = False
#     if n_cpus<=1:
#         return end_proc

#     if os.getenv("IN_MPI") is None:
#         env = os.environ.copy()
#         env.update(
#           MKL_NUM_THREADS="1",
#           OMP_NUM_THREADS="1",
#           IN_MPI="1"
#           )
#         subprocess.check_call(["mpiexec", "-n", str(n_cpus), sys.executable] +['-u']+ sys.argv, env=env)
#         end_proc = True
#     return end_proc