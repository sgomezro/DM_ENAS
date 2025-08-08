import source
from source._managerV4_5 import *
from source._helpersV4 import logSave,load_parameters
import argparse

import numpy as np
import pandas as pd
import torch
from datetime import datetime


def init_p(args):
    # p_path = 'experiments/s25_msl_100/msl_p.json'
    # p_path = 'experiments/s25_msl_300/msl_p.json'
    # p_path = 'experiments/s25_smap_300/smap_p.json'
    # p_path = 'experiments/smap_300/smap_p.json'
    # p_path = 'experiments/smap_500/smap_p.json'
    p_path = 'experiments/yahoo_300/yahoo_p.json'
    # p_path = 'experiments/msl_100/msl_p.json'
    # p_path = 'experiments/yahoo_100/yahoo_p.json'
    # p_path = 'experiments/smap_tests/smap_p.json'
    # p_path = 'experiments/shms_tests/shms_p.json'
    # p_path = 'experiments/shms/shms_p.json'
    
    p = load_parameters(p_path)

    p['maxGen'] = 10
    p['save_mod'] = 1
    p['n_workers']= args.n_processors
    p['n_slaves'] = args.n_processors-1
    p['gpus']     = []
    p['gpu_mem']  = 1000*args.gpu_mem
    
    #Creating folders directory
    folder = p['experiment_path']
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    gpus_active = False
    if len(args.gpus) > 0:
        p['gpus'] = args.gpus
    
    if args.log_save!='null':
        print('Running experiment at: '+datetime.now().strftime("%m/%d/%Y, %H:%M"))

    stdout = logSave(sys,log_save = p['experiment_path']+args.log_save)
    print('Running experiment at: '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
    return p,stdout

def lists_from_bestby_df(fname):
    df = pd.read_csv(fname)
    return df['inputs'].to_list(),df['filter'].to_list(),df['lossf'].to_list(),df['gen'].to_list()
    
def generate_p_list(args):
    l_exp = [1,2,3]
    csv_file = 'compensate_inputs'#'bestby_lossf'#
    
    p,stdout = init_p(args)
    input_dimension = 25
    if 'smap' in p['experiment_path']:
        data_type = 'smap'
        l_centers  = ['E-11']#['E-8','T-3','D-4','E-11','D-11']#['D-11','E-11','D-4','T-3']#['E-8','T-3','D-4','E-11','D-11']

    elif 'msl' in p['experiment_path']:
        data_type = 'msl'
        l_centers  = ['D-15']#['M-5','M-6','M-4','P-10','F-8']#extra['M-1','D-15','D-16']
    elif 'yahoo' in p['experiment_path']:
        data_type = 'yahoo'
        input_dimension = 1
        l_centers  = [60]#[18,60,29,50,25]#[18,60,29,50,25]

    list_p = []
    for centermost in l_centers:
        # # is_new_dataset = True
        if 's25' in p['experiment_path']:
            df_fname = f'experiments/s25_{data_type}_100/centermost_{centermost}_{csv_file}.csv'
        else:
            df_fname = f'experiments/{data_type}_100/centermost_{centermost}_{csv_file}.csv'
        l_inputs,l_filter,l_loss_f,_ = lists_from_bestby_df(df_fname)

        
        for w_size,filter,loss_f in zip(l_inputs,l_filter,l_loss_f):
            # is_new_inputs = True
            
            if any(lf in loss_f for lf in ['ap','f1']):
                diff_type = 'non-diff'
                opt_direction = 'maximize'
            elif any(lf in loss_f for lf in ['mae','ce']):
                diff_type = 'diff'
                opt_direction = 'minimize'

            for exp in l_exp:
                p_exp = p.copy()
                p_exp['seed'] = p['available_seeds'][exp-1]
                p_exp['nn_input_size'] = input_dimension
                p_exp['window_size'] = w_size
                p_exp['output_filter'] = filter
                p_exp['nn_loss_function'] = loss_f
                p_exp['opt_direction'] = opt_direction
                p_exp['experiment_root'] = p['experiment_path']

                p_exp['experiment_path'] += '{}_{}/'.format(data_type,centermost)
                if 's25' in p['experiment_path']:
                    p_exp['data_path'] += 'ad_{}_all_sensors.csv'.format(centermost)
                else:
                    p_exp['data_path'] += 'ad_{}_one_sensor.csv'.format(centermost)
                exp_name = 'exp{}_ins{}_{}_{}'.format(exp,w_size,filter,loss_f)
                p_exp['filename'] = exp_name
                p_exp['exp_print'] = '{} -> {} centermost {} window size {} filter {}-{} and seed: {}'\
                  .format(exp_name,data_type,centermost,w_size,filter,loss_f,p_exp['seed'])
                #updating output size in list of p's
                if 'target_size' in p_exp: # updating nn output size
                    p_exp['nn_output_size'] = len(p_exp['target_size'])

                list_p += [p_exp]
                    
    return list_p,stdout



def main(args):
    if args.n_processors > 1:
        if rank==0:
            list_p,stdout = generate_p_list(args)
            # caeMaster(list_p)
            cae_master(list_p,dtype=float)
            logSave(sys,stdout=stdout)
            print('Algorithm finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
        else:
            cae_worker(dtype=float)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Deep neural network centred architecture evolution'))
    parser.add_argument('-c','--n_processors', type=int, default=1,help='number of processors used to run cae')
    parser.add_argument('-g','--gpus', nargs='*', type=int, default=[], help ='Request of gpus run ENAS.')
    parser.add_argument('-ls', '--log_save', type=str,help='saving log of outputs while code is running', default='null')
    parser.add_argument('-m','--gpu_mem',type=float,help='GPU memory to be assigned to the experiment in Gb.',default=4)

    args = parser.parse_args()
    if mpi_fork(args.n_processors): sys.exit()
    main(args)