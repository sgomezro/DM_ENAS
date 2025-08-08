from source._managerV4 import *
from source._helpersV4 import logSave,load_parameters
import argparse
import numpy as np

from datetime import datetime

def init_p(args):
    p_path = 'experiments/shms/shms_p.json'
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

def generate_p_list(args):
    centremost = 27
    l_exp    = [1,2,3]
    l_gen    = ['0010','0100','0200','0300']#[str(i).zfill(4) for i in range(10,301,10)]#['0100','0110','0300']
    l_inputs = [10,20,30,50]#[10,20,30,50,100]
    l_diff_type = ['non-diff']#['diff','non-diff']
    is_ibsw = True #[True,False] #alwyas true for SHMS datasets
    arch_type = 'best'

    if centremost == 4:
        l_sensor = [4]#[1,2,3,4,29,30]#[30]
    elif centremost == 14:
        l_sensor = [14]#13,14,15,16,17,18,19]
    elif centremost == 27:
        l_sensor = [27]
    elif centremost == 35:
        l_sensor = [35]

    p,stdout = init_p(args)
    l_p = []
    for sensor in l_sensor:
        for diff_type in l_diff_type:
            is_new_dataset = True
            if diff_type == 'non-diff':
                l_filter = ['none','mio','softmax','none','mio','softmax']#['mul_none']
                l_loss_f = ['mul_f1','mul_f1','mul_f1','mul_ap','mul_ap','mul_ap']#['mlab_f1']
                opt_direction = 'maximize'
            elif diff_type == 'diff':
                l_filter = ['none','mio','softmax','mio','softmax']
                l_loss_f = ['mae','mae','mae','mul_ce','mul_ce']
                opt_direction = 'minimize'
            for exp in l_exp:
                for inputs in l_inputs:
                    is_new_inputs = True
                    for filter,loss_f in zip(l_filter,l_loss_f):
                        p_g = p.copy()
                        p_g['is_new_dataset'] = is_new_dataset
                        is_new_dataset = False
                        p_g['is_new_inputs'] = is_new_inputs
                        is_new_inputs = False
                        
                        p_g['seed'] = p['available_seeds'][exp-1]
                        p_g['nn_input_size'] = inputs
                        p_g['window_size']   = inputs
                        p_g['output_filter'] = filter
                        p_g['nn_loss_function'] = loss_f
                        p_g['opt_direction'] = opt_direction
                        p_g['cma_init_best_sw'] = is_ibsw
                        p_g['gen_list'] = l_gen

                        if centremost == 14:
                            p_g['target_size'] = ["normal","missing","saturated"]

                        exp_name = 'exp{}_ins{}_{}_{}'.format(exp,inputs,filter,loss_f,arch_type)
                        cma_folder = 'cma_IRG' #IRG Initial random guess 
                        if is_ibsw:
                            cma_folder = 'cma_IBSW' # IBSW Initial best shared-weight or

                        exp_path = '{}shms_s{}_{}/{}/'.format(p_g['experiment_path'],centremost,diff_type,exp_name)
                        p_g['arch_fname'] = '{}{}/ind_'.format(exp_path,arch_type)

                        p_g['data_path'] += 's{}_20Hz_evolution_set.csv'.format(str(centremost).zfill(2))
                        adjw_folder = '{}shms_s{}_{}/{}/{}/post_cma/'.\
                        format(p['experiment_path'],centremost,diff_type,exp_name,arch_type)
                            
                        # else:
                        #     p_g['data_path'] += 's{}_20Hz_complete_set.csv'.format(str(centremost).zfill(2))
                        #     adjw_folder = '{}shms_s{}_{}/post_adjw/completeset_s{}/{}/{}/post_cma/'.\
                        #         format(p['experiment_path'],centremost,diff_type,sensor,exp_name,arch_type)

                        p_g['adjw_folder'] = adjw_folder
                        if not os.path.exists(adjw_folder):
                            os.makedirs(adjw_folder) 

                        p_g['exp_name'] = exp_name
                        p_g['adjw_fname'] = '{}cma_'.format(adjw_folder)
                        p_g['exp_print'] = '{} -> centremost {} input size {} filter {}-{} {}'\
                          .format(exp_name,centremost,inputs,filter,loss_f,cma_folder)
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
        print('checked all architecture files exists for {}'.format(p['data_path']))
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