from source._managerV4_5 import *
from source._helpersV4 import logSave,load_parameters
import argparse
import numpy as np
import pandas as pd

from datetime import datetime

def init_p(args):
    # p_path = 'experiments/s25_msl_300/msl_p.yaml'
p_path = 'experiments/s25_smap_300/smap_p.yaml'
# p_path = 'experiments/s25_smap_100/smap_p.yaml'
# p_path = 'experiments/smap_300/smap_p.yaml'
# p_path = 'experiments/yahoo_300/yahoo_p.yaml'
# p_path = 'experiments/smap_100/msl_p.yaml'
# p_path = 'experiments/yahoo_100/yahoo_p.yaml'
# p_path = 'experiments/smap_100/smap_p.yaml'
    p = load_parameters(p_path)
    
    # p['cma_maxGen'] = 10
    p['n_workers']= args.n_processors
    p['n_slaves'] = args.n_processors-1
    p['gpus'] = []
    p['gpu_mem']  = 1000*args.gpu_mem
    
    gpus_active = False
    if len(args.gpus) > 0:
        gpus_active = True
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
    data_type,l_centers = 'smap',['E-8']#['E-8','T-3'] #['E-8','T-3','D-4','E-11','D-11'] smap dataset
    # data_type,l_centers = 'msl',['M-5'] #['M-5','M-6','M-4','P-10','F-8'] msl dataset
    # data_type,l_centers = 'yahoo', ['18','60','29','50','25']# yahoo dataset
    l_cma_files = ['bestby_lossf','compensate_inputs']
    
    l_exp    = [1,2,3]
    l_gen    = [str(i).zfill(4) for i in range(1,10,1)]+[str(i).zfill(4) for i in range(10,301,10)]
    is_ibsw = True #[True,False] #alwyas true for SHMS datasets
    arch_type = 'best'

    p,stdout = init_p(args)
    l_p = []

    for centermost in l_centers:
        if data_type == 'smap':
            if centermost == 'E-8':
                l_sensor = ['E-8','A-2','E-2','F-1','G-1']#['E-8','A-2','E-2','F-1','G-1','P-1','S-1','A-3','A-4','A-6','A-7','A-8','A-9','E-2','E-4','E-9','F-2','F-3','P-3']
            elif centermost == 'T-3':
                l_sensor = ['T-3','A-1','E-3','G-3','R-1']#['T-3','A-1','E-3','G-3','R-1','E-6','G-4','G-6','G-7']
            # elif centermost == 'D-4':
            #     l_sensor = ['D-4','B-1','D-1','E-5','G-2','P-2']#['D-4','B-1','D-1','E-5','G-2','P-2','D-3','D-5','D-6','D-7','D-8','D-9','D-12','D-13','E-7','P-4']
            # elif centermost == 'E-11':
            #     l_sensor = ['E-11','E-1','E-10','E-12','E-13','T-1','T-2']
            # elif centermost == 'D-11':
            #     l_sensor = ['D-11','A-5','D-2']
    
        elif data_type == 'msl':
            if centermost == 'M-5':
                l_sensor = ['M-5','F-7','T-4']#['M-5','F-7','T-4','M-2','M-6']
            elif centermost == 'M-4':
                l_sensor = ['M-4','D-15','F-5']#['M-4','D-15','F-5','T-13','P-11','M-1','M-3']
            elif centermost == 'P-10':
                l_sensor = ['P-10','T-9','P-15','P-14']
            elif centermost == 'F-8':
                l_sensor = ['F-8','C-1','D-14']#['F-8','C-1','D-14','C-2','M-7','S-2','T-5','T-8','T-12','D-16']
    
        elif data_type == 'yahoo':
            if centermost == '18':
                l_sensor = ['18','34','38','46','47','49','51','54','55']
            elif centermost == '60':
                l_sensor = ['60','01','02','04','05','08','09','11','12','14','16','17',\
                            '19','20','21','22','30','33','35','37','39','41','42',\
                            '43','45','48','57','62','64','65','66']
            elif centermost == '29':
                l_sensor = ['29','24','36','44','52','53','56']
            elif centermost == '50':
                l_sensor = ['50','06','07','10','13','15','26','27','28','40','59','61','63']
            elif centermost == '25':
                l_sensor = ['25','03','23','31','32','58']
        
        for cma_file in l_cma_files:
            if 's25' in p['experiment_path']:
                df_fname = f'experiments/s25_{data_type}_100/centermost_{centermost}_{cma_file}.csv'
            else:
                df_fname = f'experiments/{data_type}_100/centermost_{centermost}_{cma_file}.csv'
            l_inputs,l_filter,l_loss_f,_ = lists_from_bestby_df(df_fname)

            for sensor in l_sensor:
                is_new_dataset = True
                for w_size,filter,loss_f in zip(l_inputs,l_filter,l_loss_f):
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
                        p_g = p.copy()
                        p_g['is_new_dataset'] = is_new_dataset
                        is_new_dataset = False
                        p_g['is_new_inputs'] = is_new_inputs
                        is_new_inputs = False

                        p_g['seed'] = p['available_seeds'][exp-1]
                        p_g['nn_input_size'] = 25
                        p_g['window_size']   = w_size
                        p_g['output_filter'] = filter
                        p_g['nn_loss_function'] = loss_f
                        p_g['opt_direction'] = opt_direction
                        p_g['cma_init_best_sw'] = is_ibsw
                        p_g['gen_list'] = l_gen

                        if centermost == 14:
                            p_g['target_size'] = ["normal","missing","saturated"]

                        exp_name = 'exp{}_ins{}_{}_{}'.format(exp,w_size,filter,loss_f,arch_type)
                        cma_folder = 'cma_IRG' #IRG Initial random guess 
                        if is_ibsw:
                            cma_folder = 'cma_IBSW' # IBSW Initial best shared-weight or

                        exp_path = '{}{}_{}/{}/'.format(p_g['experiment_path'],data_type,centermost,exp_name)
                        p_g['arch_fname'] = '{}{}/ind_'.format(exp_path,arch_type)
                        if data_type == 'shms':
                            p_g['data_path'] += 's{}_20Hz_evolution_set.csv'.format(str(centermost).zfill(2))
                        else:
                            if 's25' in p['experiment_path']:
                                p_g['data_path'] += 'ad_{}_all_sensors.csv'.format(sensor)
                            else:
                                p_g['data_path'] += 'ad_{}_one_sensor.csv'.format(sensor)

                        adjw_folder = '{}{}_{}/{}/ch_{}/{}/{}/post_cma/'.\
                        format(p['experiment_path'],data_type,centermost,cma_file,sensor,exp_name,arch_type)

                        # else:
                        #     p_g['data_path'] += 's{}_20Hz_complete_set.csv'.format(str(centermost).zfill(2))
                        #     adjw_folder = '{}shms_s{}_{}/post_adjw/completeset_s{}/{}/{}/post_cma/'.\
                        #         format(p['experiment_path'],centermost,diff_type,sensor,exp_name,arch_type)

                        p_g['adjw_folder'] = adjw_folder
                        if not os.path.exists(adjw_folder):
                            os.makedirs(adjw_folder) 

                        p_g['exp_name'] = exp_name
                        p_g['adjw_fname'] = '{}cma_'.format(adjw_folder)
                        p_g['exp_print'] = '{} -> centermost {} sensor {} cma {} input size {} filter {}-{} {}'\
                          .format(exp_name,centermost,sensor,cma_file,w_size,filter,loss_f,cma_folder)
                        #updating output size in list of p's
                        if 'target_size' in p_g: # updating nn output size
                            p_g['nn_output_size'] = len(p_g['target_size'])
                        l_p += [p_g]
    return l_p,stdout

def check_if_agent_exists(p_l):
    files_missing = []
    for p in p_l:
        for g in p['gen_list']:
            arch_fname = '{}{}.npy'.format(p['arch_fname'],g)
            if not os.path.exists(arch_fname):
                files_missing += [arch_fname]
    if files_missing == []:
        print('checked all architecture files exists for the current CMA weights adjustment')
    else:
        print('List of filess missing to run cma adjustment:')
        print(files_missing)
        for item in files_missing:
            print(f' missing {item}')
        raise ValueError('Correct missing files before running cma adjustment script')
    
def main(args):
    if args.n_processors > 1:
        if rank==0: 
            list_p,stdout = generate_p_list(args)
            check_if_agent_exists(list_p)
            cma_master(list_p)
            logSave(sys,stdout=stdout)
            print('Algorithm finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
        else: 
            cma_worker()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Deep neural network centred architecture evolution'))
    parser.add_argument('-c','--n_processors', type=int, default=1,help='number of processors used to run cae')
    parser.add_argument('-g','--gpus', nargs='*', type=int, default=[], help ='Request of gpus run ENAS.')
    parser.add_argument('-ls', '--log_save', type=str,help='saving log of outputs while code is running', default='null')
    parser.add_argument('-m','--gpu_mem',type=float,help='GPU memory to be assigned to the experiment in Gb.',default=4)

    args = parser.parse_args()
    if mpi_fork(args.n_processors): sys.exit()
    main(args)