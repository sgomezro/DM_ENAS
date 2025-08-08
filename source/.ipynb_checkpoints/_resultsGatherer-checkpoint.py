import pandas as pd
import numpy as np
from ._helpersV4 import *
from .fitnessFunctionsV4 import fitness_functions
from .loaders.loaderV4 import loader
from .F.taskLossClassifierV4 import loss_classifier
from .metricsV4 import metric_class
from .timer import Timer

import zipfile
import os
import time


def get_eccd_results(l_p,save_results=False):
    file_info = pd.DataFrame({},columns=['arch_name','generation','same_arch','source_file'])
    col = ['centremost','channel','exp','inputs','filter_lossf','gen']
    t = Timer()

    folder = l_p[0]['experiment_path']+'/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder) 
            
    for init,p in enumerate(l_p):
        t.start()
        if p['is_new_dataset']:
            if init > 0:
                del dataset
                del test_input
                del test_target
            dataset = loader(p,'testset',dtype=float)
            test_input,test_target = dataset.testset()
        if p['is_new_inputs']:
            if init > 0:
                del test_input
                del test_target
            dataset.update_inputs(p['nn_input_size'])
            test_input,test_target = dataset.testset()

        general_scores = pd.DataFrame({})
        l_gen = p['gen_list']
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
            score_row_info = pd.DataFrame([[c['centremost'],c['sensor'],c['exp'],c['inputs'],c['fil_lossf'],g]],columns=col)
            
            if is_run_evaluation:
                p_exp['arch_fname'] += g+'.npy'
                p_exp['adjw_fname'] += g


                fit_test = fitness_functions(p_exp,0,set_type='testset')
                loss_f = loss_classifier(p_exp,0).loss_function()
                fit_test.set_loss_f(loss_f)
                
                fit_test.set_dataset(test_input,test_target)
                metrics = metric_class(p_exp)
                print('Calculating: {}'.format(p_exp['exp_print']+f'gen {g}'))
                predictions = fit_test.evaluate_agent_fitness(p_exp['arch_fname'],w_fname=p_exp['adjw_fname'])
                metrics.set_class_values(test_target.numpy(),predictions)
                result = metrics.get_results()
                score_report = metrics.get_multi_score_report(score_get='one_row')
                aux = pd.concat([score_row_info,score_report],axis=1)

                info = [p_exp['arch_fname'],int(g),False,p_exp['csv_fname']]
                last_source = p_exp['csv_fname']
                if save_results:
                    save_df_to_zip(result,p_exp['zip_fname'],p_exp['experiment_path'],p_exp['csv_fname'])

                #cleaning memory
                del fit_test
                del metrics
                del result
                
            else:
                print(f'Arch for g {g} same as previous. Results same as {last_source}.')
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
        if save_results:
            save_df_to_zip(file_info,p_exp['zip_fname'],p_exp['experiment_path'],'results_info',is_csv=True)
        else:
            file_info.to_csv(p_exp['zip_fname']+'_file_info.csv',index=False)
    

def is_new_gen_architecture(p,current_g,previous_g):
    # Check if current nn architecture is equal to previous generation architecture
    current_arch_file  = '{}{}.npy'.format(p['arch_fname'],current_g)
    previous_arch_file = '{}{}.npy'.format(p['arch_fname'],previous_g)

    if are_same_architectures(current_arch_file,previous_arch_file):
        return False
    return True


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
        
    col = ['centremost','channel','exp','inputs','filter_lossf','gen']
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
                        info = pd.DataFrame([[c['centremost'],channel,exp,inputs,fil_lossf,gen]],columns=col)
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
#             col = ['centremost','sensor','exp','inputs','normal_weight','filter','gen']
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
    
