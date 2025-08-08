import pandas as pd
import numpy as np
from ._helpersV4 import *
from .fitnessFunctionsV4_5 import fitness_functions
from .loaders.loaderV4_5 import loader
from .F.taskLossClassifierV4 import loss_classifier
from .metricsV4_5 import metric_class
from .timer import Timer

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import zipfile
import os, subprocess, sys
import time
import torch


def get_eccd_results(l_p,save_results=False,split_calculation=True):
    device = get_device_by_rank(l_p[0]['n_workers'],l_p[0]['gpus'],rank,verbose=True)
    file_info = pd.DataFrame({},columns=['arch_name','generation','same_arch','source_file'])
    col = ['centermost','sensor','exp','inputs','filter_lossf','gen','set_type','arch_type','cma_list']
    total_t = Timer()
    t = Timer()

    #Creating temporal folder    
    tmp_folder = l_p[0]['experiment_path']+'tmp/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder) 

    comm.bcast(l_p[0],root=0)
    group_print = ''
    
    for init,p in enumerate(l_p):
        t.start()
        set_type = p['config']['set_type']
        #Creating results files container folder

        if p['is_new_dataset']:
            dataset = None
            dataset = loader(p,set_type,dtype=float)    
            
        if p['is_new_inputs']:
            if init > 0:
                target = None
                inputs = None
                remove_tmp_worker_sets(scat_fnames)
            dataset.update_window_size(p['window_size'])
            if set_type == 'adj_w':
                inputs,target = dataset.adj_weight_set()
            elif set_type == 'testset':
                inputs,target = dataset.testset()
            elif set_type == 'trainset':
                start_idx = 100
                inputs,target = dataset.trainset_from_start_idx(start_idx)
            scat_fnames = split_input_target_sets(inputs,target,p['n_slaves'],tmp_folder)
            step = 'new_inputs'
            step = comm.bcast(step,root=0)
            comm.bcast(p,root=0)
            comm.bcast(set_type,root=0)
            comm.scatter(scat_fnames, root=0)
            
        if group_print != p['config']['exp_print_group']:
            group_print = p['config']['exp_print_group']
            print(f'\nExperiments on group: {group_print}')
        
        print('Working on {}'.format(p['config']['exp_print']))
        
        general_scores = pd.DataFrame({})
        l_gen = p['gen_list']
        loading_time = t.get_time_minutes()
        for i,g in enumerate(l_gen):
            p_exp = p.copy()
            p_exp['csv_fname'] += g

            if i == 0:
                is_run_evaluation = True
            else:
                # Check if current nn architecture is equal to previous generation architecture
                is_run_evaluation = is_new_gen_architecture(p_exp,g,l_gen[i-1])

            #to save scores
            c = p_exp['config']
            score_row_info = pd.DataFrame([[c['centermost'],c['sensor'],c['exp'],c['inputs'],c['fil_lossf'],g,c['set_type'],c['arch_type'],c['cma_list']]],columns=col)

            if is_run_evaluation:
                print(f'Calculating: generation {g}')
                if split_calculation:
                    #Calculates scores spliting the job among workers
                    step = 'calculate_cm_ap'
                    step = comm.bcast(step,root=0)
                    comm.bcast(p_exp,root=0)                
                    arch_fname = p['arch_fname'] + g + '.npy'
                    adjw_fname = p['adjw_fname'] + g
                    comm.bcast((arch_fname,adjw_fname), root=0)
                    gather_predictions = comm.gather(0,root=0)
                    predictions = np.vstack(gather_predictions[1:])
                    pred_time = t.get_time_minutes()-loading_time
                    
                    
                    metrics = metric_class(p_exp)
                    # metrics.set_class_values(target.numpy(),sub_predictions)
                    # result = metrics.get_results()
    
                    gather_cm_ap = comm.gather(0,root=0)
                    score_report = metrics.ensamble_score_report(gather_cm_ap[1:],'confusion_matrix')
                else:
                    #calculates scores directly, workers only calculate predictions
                    step = 'calculate_single_report'
                    step = comm.bcast(step,root=0)
                    comm.bcast(p_exp,root=0)                
                    arch_fname = p['arch_fname'] + g + '.npy'
                    adjw_fname = p['adjw_fname'] + g
                    comm.bcast((arch_fname,adjw_fname), root=0)
                    gather_predictions = comm.gather(0,root=0)
                    predictions = np.vstack(gather_predictions[1:])
                    pred_time = t.get_time_minutes()-loading_time
                    
                    metrics = metric_class(p_exp,device)
                    # metrics.set_class_values(target,predictions)
                    # score_report = metrics.get_score_report()
                    gather_report = comm.gather(0,root=0)
                    print(f'report values {gather_report[1]}')
                    score_report = metrics.ensamble_score_report(gather_report[1],'single_report')

                score_report['load_time'] = loading_time
                score_report['pred_time'] = pred_time
                score_report['scoring_time']= t.get_time_minutes()-(loading_time+pred_time)
                aux = pd.concat([score_row_info,score_report],axis=1)

                info = [p_exp['arch_fname'],int(g),False,p_exp['csv_fname']]
                last_source = p_exp['csv_fname']
                if save_results:
                    save_df_to_zip(result,p_exp['zip_fname'],p_exp['experiment_path'],p_exp['csv_fname'])

                #cleaning memory
                metrics = None
                gather_predictions = None
                gather_cm_ap = None
                gather_report = None
                result = None
                
            else:
                print(f'generaion {g} same as {last_source}.')
                info = [p_exp['arch_fname'],int(g),True,last_source]
                aux = pd.concat([score_row_info,score_report],axis=1)

            #generating score report
            general_scores = pd.concat([general_scores,aux],axis=0)
            #generating info file about results
            file_info.loc[-1] = info
            file_info.index = file_info.index + 1
            file_info = file_info.sort_values('generation')
            file_info.reset_index(drop=True,inplace=True)
           
            #saving score report
            general_scores.to_csv(p_exp['zip_fname']+'.csv',index=False)

        t.stop()
        # if save_results:
        #     save_df_to_zip(file_info,p_exp['zip_fname'],p_exp['experiment_path'],'results_info',is_csv=True)
        # else:
        #     file_info.to_csv(p_exp['zip_fname']+'_file_info.csv',index=False)

    step = 'stop_workers'
    step = comm.bcast(step, root=0) #stops all workers
    remove_tmp_worker_sets(scat_fnames)
    
def worker(batch_size=0):
    p = comm.bcast(None,root=0)
    device = get_device_by_rank(p['n_workers'],p['gpus'],rank,verbose=True)
    
    while True:
        step = comm.bcast(None,root=0)
        if step == 'new_inputs':
            p = comm.bcast(None,root=0)
            set_type = comm.bcast(None,root=0)
            inputs = None
            target = None
            inputs_fname,target_fname = comm.scatter(None, root=0)

            # Loading the data
            inputs = torch.from_numpy(np.load(inputs_fname))
            target = torch.from_numpy(np.load(target_fname))

            #generating the fitness function
            # fit = fitness_functions(p,rank,set_type=set_type)
            loss_f = loss_classifier(p,rank).loss_function()
            # fit.set_loss_f(loss_f)
            # fit.set_dataset(inputs,target)

        elif step == 'calculate_cm_ap':
            p = comm.bcast(None,root=0)
            arch_fname,adjw_fname = comm.bcast(None,root=0)
            fit = fitness_functions(p,rank,set_type=set_type)
            fit.set_loss_f(loss_f)
            fit.set_dataset(inputs.clone(),target)
            sub_predictions = fit.evaluate_agent_fitness(arch_fname,w_fname=adjw_fname).cpu()
            comm.gather(sub_predictions,root=0)

            metrics = metric_class(p,device)
            metrics.set_class_values(target.clone(),sub_predictions)
            sub_cm_ap = metrics.get_confusion_matrix()
            comm.gather(sub_cm_ap,root=0)
            
            #cleaning variables
            fit = None
            sub_predictions = None
            sub_scores = None
            metrics = None
            torch.cuda.empty_cache()

        elif step == 'calculate_single_report':
            p = comm.bcast(None,root=0)
            arch_fname,adjw_fname = comm.bcast(None,root=0)
            fit = fitness_functions(p,rank,set_type=set_type)
            fit.set_loss_f(loss_f)
            fit.set_dataset(inputs.clone(),target)
            predictions = fit.evaluate_agent_fitness(arch_fname,w_fname=adjw_fname).cpu()
            comm.gather(predictions,root=0)

            metrics = metric_class(p,device)
            metrics.set_class_values(target.clone(),predictions)
            sub_report = metrics.get_score_report()
            comm.gather(sub_report,root=0)
            
            #cleaning variables
            fit = None
            predictions = None
            sub_report = None
            metrics = None
            torch.cuda.empty_cache()

        elif step == 'stop_workers':
            break


    
def save_df_to_zip(df, zip_file_path, temp_path, csv_file_name,is_csv=False):
    # Convert DataFrame to CSV
    if is_csv:
        temp_csv_fname = f'{temp_path}/tmp/{int(time.time() * 1000)}.csv'
        csv_file_name += '.csv'
        df.to_csv(temp_csv_fname, index=False)
    else:
        temp_csv_fname = f'{temp_path}/tmp/{int(time.time() * 1000)}.pkl.gz'
        csv_file_name += '.pkl.gz'
        df.to_pickle(temp_csv_fname, compression="gzip")

    # Open or create the ZIP file and append the CSV file
    with zipfile.ZipFile(zip_file_path+'.zip', mode='a', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_csv_fname, arcname=csv_file_name)

    # Remove the temporary CSV file
    os.remove(temp_csv_fname)

def calculate_scores(config):
    cat_scores = pd.DataFrame({})
    general_scores = pd.DataFrame({})


def get_scores(c,p):
    #Creating cma output files container folder
    tmp_folder = c['scores_path']+'/tmp/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        
    col = ['centermost','channel','exp','inputs','filter_lossf','gen']
    general_scores = pd.DataFrame({})
    
    for exp in c['exp_list']:
        for channel in c['channel_list']:
            for inputs in c['inputs_list']:
                for filter,lossf in zip(c['filter_list'],c['loss_f_list']):
                    fil_lossf= f'{filter}_{lossf}'
                    exp_name = 'exp{}_ins{}_{}'.format(exp,inputs,fil_lossf)
                    zip_fname = '{}{}.zip'.format(c['results_path'],exp_name)
                    csv_fname = '{}_g'.format(exp_name)
                    
                    # Extract the  CSV file
                    with zipfile.ZipFile(zip_fname, 'r') as zip_ref:
                        zip_ref.extract('results_info.csv', c['scores_path']+'/tmp/')
                    
                    results_info = pd.read_csv(c['scores_path']+'tmp/results_info.csv')
                    
                    same_arch = results_info.loc[:,'same_arch'].values
                    for g,gen in enumerate(results_info.loc[:,'generation'].values):
                        info = pd.DataFrame([[c['centermost'],channel,exp,inputs,fil_lossf,gen]],columns=col)
                        if not same_arch[g]:
                            fname = '{}{}.pkl.gz'.format(csv_fname,str(gen).zfill(4))
                            ftemp = '{}tmp/'.format(c['scores_path']) 
                            metrics = metric_class(p)
                            metrics.set_from_zip(fname,ftemp,zip_fname)
                            row_report = metrics.get_multi_score_report(score_get='one_row')
                        aux = pd.concat([info,row_report],axis=1)
                        general_scores = pd.concat([general_scores,aux],axis=0)
    
    general_scores.reset_index(inplace=True,drop=True)
    return general_scores

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

# def get_ECCD_scores(cl_sensor,l_sensor,end_dir,l_cat,n_rows=None):
#     storage_path = '/storage_1/aDetection'
#     p_adjust = 'experiments/aDetection_s{}_{}/aDetection.json'.format(cl_sensor,end_dir)
#     p = loadParameters(p_adjust)

#     l_exp  = [1,2,3,4,5]
#     inputs = p['best_inputs']
#     nW     = p['best_nW']
#     filter = p['best_filter']
#     gen    = str(p['best_gen']).zfill(4)
#     p['nn_input_size'] = inputs
#     p['target_size'] =  l_cat

#     cat_scores = pd.DataFrame({})
#     general_scores = pd.DataFrame({})
#     for sensor in l_sensor:
#         for exp in l_exp:
#             col = ['centermost','sensor','exp','inputs','normal_weight','filter','gen']
#             info = [cl_sensor,sensor,exp,inputs,nW,filter,gen]
#             head = pd.DataFrame([info],columns=col)

#             c_name = 'result_s{}exp{}in{}nW{}g{}{}.csv'.format(sensor,exp,inputs,str(int(nW*100)).zfill(4),filter,gen)
#             filename = '{}/experiments/s{}_{}/best_results/ECCD_s{}_{}'.format(storage_path,cl_sensor,end_dir,cl_sensor,c_name)
#             metrics = Metrics(p)
#             metrics.set_from_csv(filename,n_rows=n_rows)
#             cat_rep = metrics.get_multi_score_report(score_get='full').T
#             cat_rep.index.name='category'

#             cat_report =pd.DataFrame({})
#             columns = ['precision','recall','f1-score']
#             cat_size = len(metrics.target_labels)
#             aux = pd.merge(head.copy(),cat_rep.iloc[:cat_size,:3].reset_index(),how='cross')
#             cat_report = pd.concat([cat_report,aux])
#             cat_scores = pd.concat([cat_scores,cat_report])

#             if end_dir == 'bin':
#                 gen_report = metrics.get_bin_score_report(score_get='one_row')
#             elif (end_dir == 'filter') | (end_dir == 'weightned'):
#                 gen_report = metrics.get_multi_score_report(score_get='one_row')
#             else:
#                 raise ValueError('It is neccesary to define end directory as bin, filter, or weightned')
#             aux2 = pd.merge(head.copy(),gen_report,how='cross')
#             general_scores = pd.concat([general_scores,aux2],axis=0)
#             del metrics

#     cat_scores.reset_index(inplace=True,drop=True)
#     general_scores.reset_index(inplace=True,drop=True)

#     path = 'experiments/aDetection_s{}_{}/score_results/'.format(cl_sensor,end_dir)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     prefix = 'general_scores'
#     filename = 's{}_cluster_{}.csv'.format(cl_sensor,prefix)
#     general_scores.to_csv(path+filename)

#     prefix = 'category_scores'
#     filename = 's{}_cluster_{}.csv'.format(cl_sensor,prefix)
#     cat_scores.to_csv(path+filename)
    

