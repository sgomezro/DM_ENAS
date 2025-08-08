from .._helpersV3_2 import get_device_by_rank
# genral loss function class caller
def task_loss_c(p,rank=0):
    # Setting cpu or gpu device
    device = get_device_by_rank(p['n_workers'],p['gpus'],rank,verbose=False)

    if (p['task_type'] == 'binary_classification'):  #| (p['task'] == 'ad_smap'):
        from .bin_classifier_f import bin_class as loss_c
    elif p['task_type'] == 'multi_classification':
        from .multi_classifier import multi_class as loss_c
    else:
        raise Exception('Classifier type must be defined. Current selection {} not supported'.format(p['task']))
    
    return loss_c(p,device=device)
        
        

