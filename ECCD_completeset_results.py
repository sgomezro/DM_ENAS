from source._resultsGathererV4_2 import *
from source._helpersV4 import logSave,load_parameters

import argparse
import os,sys
from datetime import datetime

def initialize(args):
    
    p_path = 'experiments/shms/shms_p.json'
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
    centremost = 27
    df_fname = f'evolset_s{centremost}_bestby_lossf.csv'
    is_evolset = False
    arch_type = 'pre_adjw'
    post_cma = 'post_cma'#'cma'
    
    l_exp = [1,2,3]#[1,2,3]

    if centremost == 4:
        l_sensor = [1,2,3,4,29,30]#[30]
    elif centremost == 14:
        l_sensor = [13,14,15,16,17,18,19]
    elif centremost == 27:
        l_sensor = [27]#[25,26,27,28]
    elif centremost == 35:
        l_sensor = [33,34,35]
    l_gen = ['0300']
    # l_inputs = [10,10]
    # l_filter = ['mio','mio']
    # l_loss_f = ['mul_ap','mul_f1']
    # # l_diff_type = ['non-diff','non_diff']
    
    p,stdout = initialize(args)
    l_inputs,l_filter,l_loss_f,_ = lists_from_bestby_df(p['experiment_path']+df_fname)

    l_p = []
    for sensor in l_sensor:
        is_new_dataset = True
        for inputs,filter,loss_f in zip(l_inputs,l_filter,l_loss_f):
            is_new_inputs = True
            # if diff_type == 'non-diff':
            if any(lf in loss_f for lf in ['ap','f1']):
                diff_type = 'non-diff'
                opt_direction = 'maximize'
            # elif diff_type == 'diff':
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
                config['centremost'] = centremost
                config['sensor'] = sensor
                config['exp'] = exp
                config['inputs'] = inputs
                config['fil_lossf'] = '{}_{}'.format(filter,loss_f)
                p_exp['config'] = config

                if centremost == 14:
                    p_exp['target_size'] = ["normal","missing","saturated"]

                exp_name = 'exp{}_ins{}_{}_{}'.format(exp,inputs,filter,loss_f,arch_type)
                exp_path = '{}shms_s{}_{}/{}/'.format(p_exp['experiment_path'],centremost,diff_type,exp_name)
                if is_evolset:
                    p_exp['data_path'] += 's{}_20Hz_evolution_set.csv'.format(str(centremost).zfill(2))
                    save_folder = 'results/evolset_s{}/{}/{}'.format(centremost,arch_type,post_cma)
                else:
                    p_exp['data_path'] += 's{}_20Hz_complete_set.csv'.format(str(centremost).zfill(2))
                    save_folder = 'results/completeset_s{}/{}/{}'.format(sensor,arch_type,post_cma)
                    
                p_exp['save_folder'] = save_folder
                results_folder = '{}/shms_s{}_{}/{}'.format(p['experiment_path'],centremost,diff_type,save_folder)
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder) 
                
                p_exp['arch_fname'] = '{}{}/ind_'.format(exp_path,arch_type)
                p_exp['adjw_fname'] = '{}{}/{}/cma_'.\
                    format(exp_path,arch_type,post_cma)
                p_exp['csv_fname']  = '{}_g'.format(exp_name)
                p_exp['zip_fname']  = '{}shms_s{}_{}/{}/{}'.\
                    format(p['experiment_path'],centremost,diff_type,save_folder,exp_name)
                p_exp['exp_print'] = '{} -> centremost {} input size {} filter {}-{} '.\
                    format(exp_name,centremost,inputs,filter,loss_f)
                #updating output size in list of p's
                if 'target_size' in p_exp: # updating nn output size
                    p_exp['nn_output_size'] = len(p_exp['target_size'])
                l_p += [p_exp]

    return l_p,stdout


def main(args):
    if args.n_processors > 1:
        if rank == 0:
            list_p,stdout = generate_p_list(args)
            get_eccd_results(list_p,save_results=False)
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