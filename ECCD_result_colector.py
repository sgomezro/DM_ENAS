from source._resultsGatherer import *
from source._helpersV4 import logSave,load_parameters

import argparse
import os,sys
from datetime import datetime

def initialize(args):
    channel = 4
    testing = True
    diff_type = 'diff'
    arch_type = 'pre_adjw'
    
    p_path = 'experiments/shms/shms_p.yaml'
    p = loadParameters(p_path)

    save_filename = 'ECCD_ch{}_'.format(channel)
    save_folder = 'results/{}'.format(arch_type)
    p['save_folder'] = save_folder
    p['save_filename'] = save_filename

    #Creating cma output files container folder
    folder = f'{p['storage_path']}/shms_s{channel}_{diff_type}/{save_folder}'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    stdout = logSave(sys,log_save = p['experiment_path']+args.log_save)
    print('Running experiment at: '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
    return p,stdout,testing

def generate_p_list(args):
#     l_sensor = [1,2,3,4,29,30]
#     l_sensor = [13,14,15,16,17,18,19]
#    l_sensor = [25,26,27,28]
    l_sensor = [4]#[33,34,35]
    
    l_exp  = [1]#[1,2,3,4,5]
   
    l_p = []
    exp_info = []
    p,stdout,testing = initialize(args)
    # nW     = p['best_nW']
    gen    = str(300]).zfill(4)
    inputs = 10
    filter = p['best_filter']
    
    for sensor in l_sensor:
        for exp in l_exp:
            p_g = p.copy()
            p_g['sensor'] = sensor
            p_g['nn_input_size'] = inputs
            p_g['output_filter'] = filter
            if p['classification_type'] == 'multilabel':
                p_g['out_w_vector'] = [nW,1,1,1]
                exp_name = 'exp{}_ins{}_nW{}_{}'.format(exp,inputs,str(int(nW*100)).zfill(4),filter)
            elif p['classification_type'] == 'binary':
                p_g['out_w_vector'] = [nW,1]
                exp_name = 'bin_exp{}_ins{}_nW{}_{}'.format(exp,inputs,str(int(nW*100)).zfill(4),filter)

            p_g['filename'] = '{}{}_best/ind_{}.npy'.format(p_g['experiment_path'],exp_name,gen)
            p_g['cma_weights'] = '{}{}_best/cma_{}_wn{}'.format(p_g['experiment_path'],exp_name,gen,int(nW*100))
            info = [exp,inputs,nW,filter,gen,exp_name]

            
            l_p += [p_g]
            exp_info += [info]
    return l_p,exp_info,e_gpu,orig_stdout,testing


def main(args):
        list_p,exp_info,e_gpu,orig_stdout,testing = generate_p_list(args)
        colect_single_processor(list_p,exp_info,testing=testing)
        logSave(sys,stdout=orig_stdout)
        print('ECCD colector finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))

#     elif args.n_procs > 1:
#         if rank==0: 
#             list_p,exp_info,e_gpu,orig_stdout = generate_p_list(args)
#             print('Running with {} workers and {} gpus'.format(list_p[0]['n_workers'],list_p[0]['n_gpus'] if e_gpu else 0))
#             master_colector(list_p,exp_info,testing=True)
#             print('Algorithm finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
#         else: 
#             slave_colector()


#     if rank == 0:
#         logSave(sys,stdout=orig_stdout)
#         print('ECCD colector finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Deep neural network centred architecture evolution'))
    parser.add_argument('-c','--n_processors', type=int, default=1,help='number of processors used to run cae')
    parser.add_argument('-g','--gpus', nargs='*', type=int, default=[], help ='Request of gpus run ENAS.')
    parser.add_argument('-ls', '--log_save', type=str,help='saving log of outputs while code is running', default='null')
    parser.add_argument('-m','--gpu_mem',type=float,help='GPU memory to be assigned to the experiment in Gb.',default=4)

    args = parser.parse_args()
    main(args)