from source._colector_manager import *
from source._helpers import *

import argparse
import os,sys
from datetime import datetime

def initialize(args):
    cl_sensor = 35
    end_dir = 'weightned'
    n_rows = None
#     n_rows = 10**3

#     l_sensor = [1,2,3,4,29,30]
#     l_sensor = [13,14,15,16,17,18,19]
#     l_sensor = [25,26,27,28]
    l_sensor = [33,34,35]
    
    #categories to measure
    if end_dir == 'bin':
        l_cat = ['normal','anomaly']
    elif (end_dir == 'filter') | (end_dir == 'weightned'):
        l_cat = ['normal','missing','saturated','square']
    #     l_cat = ['normal','missing','saturated']


    log_save = 'experiments/aDetection_s{}_{}/{}'.format(cl_sensor,end_dir,args.log_save)
    orig_stdout = logSave(sys,log_save = log_save)
    print('Running experiment at: '+datetime.now().strftime("%m/%d/%Y, %H:%M"))
    return [cl_sensor,l_sensor,end_dir,l_cat,n_rows],orig_stdout


def main(args):
        info,orig_stdout = initialize(args)
        get_ECCD_scores(info[0],info[1],info[2],info[3],n_rows=info[4])
        logSave(sys,stdout=orig_stdout)
        print('ECCD colector finished at '+datetime.now().strftime("%m/%d/%Y, %H:%M"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('ECCD results colector'))
    parser.add_argument('-ls', '--log_save', type=str,help='saving log of outputs while code is running', default='null')
    args = parser.parse_args()

    main(args)