from source._resultsGathererV4_3 import *
from source._helpersV4 import logSave,load_parameters

import pandas as pd
import numpy as np

import argparse
import os,sys
from datetime import datetime

def initialize(args):
    # p_path = 'experiments/yahoo_100/yahoo_p.json'
    p_path = 'experiments/smap_100/smap_p.json'
    # p_path = 'experiments/smap_tests/smap_p.json'
    # p_path = 'experiments/shms/shms_p.json'
    p = load_parameters(p_path)
    p['n_workers']= args.n_processors
    p['n_slaves'] = args.n_processors-1
    p['gpus']     = []
    p['gpu_mem']  = 1000*args.gpu_mem

    gpus_active = False
    if len(args.gpus) > 0:
        p['gpus'] = args.gpus
    
    stdout = logSave(sys,log_save = p['experiment_path']+args.log_save)
    print('Running experiment at: '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
    return p,stdout

def lists_from_bestby_df(fname):
    df = pd.read_csv(fname)
    return df['inputs'].to_list(),df['filter'].to_list(),df['lossf'].to_list(),df['gen'].to_list()

def generate_p_list(args):
    centermost = 'D-11' #['E-8','T-3','D-4','E-11','D-11'] smap dataset
    # centermost = '50' # [18,60,29,50,25] yahoo dataset
    is_evolset = False
    arch_type = 'best'
    data_type = 'smap'
    set_type  = 'testset'
    
    df_fname = f'centermost_{centermost}_bestby_lossf.csv'
    l_exp    = [1,2,3]
    l_gen    = [str(i).zfill(4) for i in range(10,101,10)]#['0100','0110','0300']
    is_ibsw = True #[True,False] #alwyas true for SHMS datasets

    if data_type == 'smap':
        if centermost == 'E-8':
            l_channel = ['E-8','A-2','A-3','A-4','A-6','A-7','A-8','A-9','E-2',\
                        'E-4','E-9','F-1','F-2','F-3','G-1','P-1','P-3','S-1']
        elif centermost == 'T-3':
            l_channel = ['T-3','A-1','E-3','E-6','G-3','G-4','G-6','G-7','R-1']
        elif centermost == 'D-4':
            l_channel = ['D-4','B-1','D-1','D-3','D-5','D-6','D-7','D-8','D-9',\
                        'D-12','D-13','E-5','E-7','G-2','P-2','P-4']
        elif centermost == 'E-11':
            l_channel = ['E-11','E-1','E-10','E-12','E-13','T-1','T-2']
        elif centermost == 'D-11':
            l_channel = ['D-11','A-5','D-2']

    elif data_type == 'yahoo':
        if centermost == '18':
            l_channel = ['18','34','38','46','47','49','51','54','55']
        elif centermost == '60':
            l_channel = ['60','01','02','04','05','08','09','11','12','14','16','17',\
                        '19','20','21','22','30','33','35','37','39','41','42',\
                        '43','45','48','57','62','64','65','66']
        elif centermost == '29':
            l_channel = ['29','24','36','44','52','53','56']
        elif centermost == '50':
            l_channel = ['50','06','07','10','13','15','26','27','28','40','59','61','63']
        elif centermost == '25':
            l_channel = ['25','03','23','31','32','58']    
    
    
    p,stdout = initialize(args)
    l_inputs,l_filter,l_loss_f,_ = lists_from_bestby_df(p['experiment_path']+df_fname)
    l_p = []
    for channel in l_channel:
        is_new_dataset = True

        save_folder = 'results/{}_set/channel_{}/{}'.format(set_type,channel,arch_type)
        p['save_folder'] = save_folder
        folder = '{}/{}_{}/{}'.format(p['experiment_path'],data_type,centermost,save_folder)
        if not os.path.exists(folder):
            os.makedirs(folder)    

        for inputs,filter,loss_f in zip(l_inputs,l_filter,l_loss_f):
            is_new_inputs = True
            
            if any(lf in loss_f for lf in ['ap','f1']):
                diff_type = 'non-diff'
                opt_direction = 'maximize'
            elif any(lf in loss_f for lf in ['mae','ce']):
                diff_type = 'diff'
                opt_direction = 'minimize'

            for exp in l_exp:
                p_exp = p.copy()
                config = {}
                p_exp['is_new_dataset'] = is_new_dataset
                is_new_dataset = False
                p_exp['is_new_inputs'] = is_new_inputs
                is_new_inputs = False
                p_exp['gen_list'] = l_gen
                p_exp['seed'] = p['available_seeds'][exp-1]
                p_exp['nn_input_size'] = inputs
                p_exp['window_size']   = inputs
                p_exp['output_filter'] = filter
                p_exp['nn_loss_function'] = loss_f
                p_exp['opt_direction'] = opt_direction
                config['centremost'] = centermost
                config['sensor'] = channel
                config['exp'] = exp
                config['inputs'] = inputs
                config['fil_lossf'] = '{}_{}'.format(filter,loss_f)
                config['set_type'] = set_type
                p_exp['config'] = config

                exp_name = 'exp{}_ins{}_{}_{}'.format(exp,inputs,filter,loss_f,arch_type)
                exp_path = '{}{}_{}/{}/'.format(p_exp['experiment_path'],data_type,centermost,exp_name)
                p_exp['data_path'] += 'ad_{}_one_sensor.csv'.format(channel)
                p_exp['arch_fname'] = '{}{}/ind_'.format(exp_path,arch_type)

                p_exp['adjw_fname'] = '{}{}_{}/best_flf/ch_{}/{}/{}/post_cma/cma_'.\
                format(p['experiment_path'],data_type,centermost,channel,exp_name,arch_type)
                p_exp['csv_fname']  = '{}_g'.format(exp_name)
                p_exp['zip_fname']  = '{}{}_{}/{}/{}'.\
                    format(p['experiment_path'],data_type,centermost,save_folder,exp_name)
                p_exp['exp_print'] = '{} -> centermost {} channel {} input size {} filter {}-{} '.\
                    format(exp_name,centermost,channel,inputs,filter,loss_f)
                #updating output size in list of p's
                if 'target_size' in p_exp: # updating nn output size
                    p_exp['nn_output_size'] = len(p_exp['target_size'])
                l_p += [p_exp]

    return l_p,stdout


def main(args):
    if args.n_processors > 1:
        if rank == 0:
            list_p,stdout = generate_p_list(args)
            get_eccd_results(list_p,save_results=False,split_calculation=False)
            logSave(sys,stdout=stdout)
            print('ECCD colector finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))

        else:
            worker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Deep neural network centred architecture evolution'))
    parser.add_argument('-c','--n_processors', type=int, default=1,help='number of processors used to run cae')
    parser.add_argument('-g','--gpus', nargs='*', type=int, default=[], help ='Request of gpus run ENAS.')
    parser.add_argument('-ls', '--log_save', type=str,help='saving log of outputs while code is running', default='null')
    parser.add_argument('-m','--gpu_mem',type=float,help='GPU memory to be assigned to the experiment in Gb.',default=4)

    args = parser.parse_args()
    if mpi_fork(args.n_processors): sys.exit()
    main(args)