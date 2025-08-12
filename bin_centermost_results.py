from source._resultsGathererV4_5 import *
from source._helpersV4 import logSave,load_parameters

import argparse
import os,sys
from datetime import datetime

def initialize(args):
    p_path = 'experiments/s25_msl_100/msl_p.yaml'
# p_path = 'experiments/s25_smap_100/smap_p.yaml'
# p_path = 'experiments/msl_100/msl_p.yaml'
# p_path = 'experiments/yahoo_100/yahoo_p.yaml'
# p_path = 'experiments/smap_100/smap_p.yaml'
# p_path = 'experiments/smap_tests/smap_p.yaml'
# p_path = 'experiments/shms/shms_p.yaml'
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

def generate_p_list(args):
    centermost = 'D-16' #['M-5','M-6','M-4','P-10','F-8'] msl dataset
    # centermost = 'T-3' #['E-8','T-3','D-4','E-11','D-11'] smap dataset
    # centermost = '50' # [18,60,29,50,25] yahoo dataset
    is_evolset = True
    arch_type = 'pre_adjw'
    data_type = 'msl'
    set_type = 'adj_w'
    
    l_diff_type = ['non-diff']#['diff','non-diff']
    l_exp = [1,2,3]
    # l_inputs = [10,20,30,50,100]
    l_inputs  = [2,5,10,20,30,50]
    l_gen = ['0100']#[str(i).zfill(4) for i in range(10,101,10)]
    
    p,stdout = initialize(args)
    save_folder = 'results/{}_set/channel_{}/{}'.format(set_type,centermost,arch_type)
    p['save_folder'] = save_folder
    
    #Creating cma output files container folder
    folder = '{}/{}_{}/{}'.format(p['experiment_path'],data_type,centermost,save_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)    
    
    l_p = []
    is_new_dataset = True
    for diff_type in l_diff_type:
        if diff_type == 'non-diff':
            l_filter = ['mio','softmax']#['none','mio','softmax','none','mio','softmax']#['mul_none']
            l_loss_f = ['bin_ap','bin_ap']#['bin_f1','bin_f1','bin_f1','bin_ap','bin_ap','bin_ap']#['mlab_f1']
            opt_direction = 'maximize'
        elif diff_type == 'diff':
            l_filter = ['none','mio','softmax','mio','softmax']
            l_loss_f = ['mae','mae','mae','bin_ce','bin_ce']
            opt_direction = 'minimize'
        
        for w_size in l_inputs:
            is_new_inputs = True
            for exp in l_exp:
                for filter,loss_f in zip(l_filter,l_loss_f):
                    p_exp = p.copy()
                    config = {}
                    p_exp['is_new_dataset'] = is_new_dataset
                    is_new_dataset = False
                    p_exp['is_new_inputs'] = is_new_inputs
                    is_new_inputs = False
                    p_exp['gen_list'] = l_gen
                    p_exp['seed'] = p['available_seeds'][exp-1]
                    p_exp['nn_input_size'] = 25
                    p_exp['window_size']   = w_size
                    p_exp['output_filter'] = filter
                    p_exp['nn_loss_function'] = loss_f
                    p_exp['opt_direction'] = opt_direction
                    config['centermost'] = centermost
                    config['sensor'] = centermost
                    config['exp'] = exp
                    config['inputs'] = w_size
                    config['fil_lossf'] = '{}_{}'.format(filter,loss_f)
                    config['set_type'] = set_type
    
                    exp_name = 'exp{}_ins{}_{}_{}'.format(exp,w_size,filter,loss_f,arch_type)
                    exp_path = '{}{}_{}/{}/'.format(p_exp['experiment_path'],data_type,centermost,exp_name)
                    if 's25' in p['experiment_path']:
                        p_exp['data_path'] += 'ad_{}_all_sensors.csv'.format(centermost)    
                    else:
                        p_exp['data_path'] += 'ad_{}_one_sensor.csv'.format(centermost)
                    
                    p_exp['arch_fname'] = '{}{}/ind_'.format(exp_path,arch_type)
                    p_exp['adjw_fname'] = '{}{}/cma/cma_'.format(exp_path,arch_type)
                    p_exp['csv_fname']  = '{}_g'.format(exp_name)
                    p_exp['zip_fname']  = '{}{}_{}/{}/{}'.\
                        format(p['experiment_path'],data_type,centermost,save_folder,exp_name)
                    config['set_type'] = set_type
                    config['arch_type']= arch_type
                    config['cma_list'] = 'None'
                    config['exp_print_group'] = 'centermost {} sensor {}, arch {}, set {}'.\
                                        format(centermost,centermost,arch_type,set_type)
                    config['exp_print'] = '{} -> window size {} filter {}-{} '.\
                                        format(exp_name,w_size,filter,loss_f)
                    p_exp['config'] = config
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