import source
from source._managerV4 import *
from source._helpersV4 import logSave,load_parameters
import argparse

import numpy as np
import torch
from datetime import datetime


def init_p(args):
    p_path = 'experiments/msl_100/msl_p.json'
    # p_path = 'experiments/yahoo_100/yahoo_p.json'
    # p_path = 'experiments/smap_tests/smap_p.json'
    # p_path = 'experiments/shms_tests/shms_p.json'
    # p_path = 'experiments/shms/shms_p.json'
    
    p = load_parameters(p_path)

    # p['maxGen'] = 10
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

def generate_p_list(args):
    l_exp    = [1,2,3]
    l_diff_type = ['diff','non-diff']
    l_inputs  = [10,20,30,50,100]

        
    p,stdout = init_p(args)
    if 'smap' in p['experiment_path']:
        dataset = 'smap'
        l_centers  = ['E-8','T-3','D-4','E-11','D-11']
    elif 'msl' in p['experiment_path']:
        dataset = 'msl'
        l_centers  = ['M-5','M-4','P-10','F-8','M-6']
    elif 'yahoo' in p['experiment_path']:
        dataset = 'yahoo'
        l_centers  = [18,60,29,50,25]
        
    list_p = []
    for centermost in l_centers:
        for diff_type in l_diff_type:
            if diff_type == 'non-diff':
                l_filter = ['none','mio','softmax','none','mio','softmax']
                l_loss_f = ['bin_f1','bin_f1','bin_f1','bin_ap','bin_ap','bin_ap']
                opt_direction = 'maximize'
            elif diff_type == 'diff':
                l_filter = ['none','mio','softmax','mio','softmax']
                l_loss_f = ['mae','mae','mae','bin_ce','bin_ce']
                opt_direction = 'minimize'

            for exp in l_exp:
                for inputs in l_inputs:
                    for filter,loss_f in zip(l_filter,l_loss_f):
                        p_exp = p.copy()
                        p_exp['seed'] = p['available_seeds'][exp-1]
                        p_exp['nn_input_size'] = inputs
                        p_exp['window_size'] = inputs
                        p_exp['output_filter'] = filter
                        p_exp['nn_loss_function'] = loss_f
                        p_exp['opt_direction'] = opt_direction

                        p_exp['experiment_path'] += '{}_{}/'.format(dataset,centermost)
                        p_exp['data_path'] += 'ad_{}_one_sensor.csv'.format(centermost) 
                        exp_name = 'exp{}_ins{}_{}_{}'.format(exp,inputs,filter,loss_f)
                        p_exp['filename'] = exp_name
                        p_exp['exp_print'] = '{} -> {} centermost {} input size {} filter {}-{} and seed: {}'\
                          .format(exp_name,dataset,centermost,inputs,filter,loss_f,p_exp['seed'])
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