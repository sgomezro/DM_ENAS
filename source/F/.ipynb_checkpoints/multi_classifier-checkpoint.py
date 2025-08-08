import numpy as np
import torch
import warnings
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, average_precision_score,log_loss




class multi_class():
    def __init__ (self,p,device,dtype=torch.float32):
        if p['nn_output_size'] < 2:
            raise OtherException('Type of class function selected is for multi classification, nevertheless, binary-class output size is selected.')
        # Setting cpu or gpu device
        self.device = device
        self.dtype  = dtype

        # Setting NN loss function
        self.loss_f = p['nn_loss_function']
        
        # Setting weights vector parameter
        if 'out_w_vector' in p:
            if ('mae' in self.loss_f) | ('ce' in self.loss_f):
                w_vector = p['out_w_vector']
            else:
                raise Exception('weigth in multi classes supported only for mae and ce loss functions.')
                # w_vector = np.ones(p['nn_output_size'])
        else:
            w_vector = np.ones(p['nn_output_size'])
        self.w_vector = torch.Tensor(w_vector)
        

        
        # Setting output filter
        if 'output_filter' in p:
            self.filter_type = p['output_filter']
        else:
            self.filter_type = None

    def loss_function(self):
        if self.loss_f == 'mae':
            def fun_error(target,output):
                return mean_absolute_error(target,output,self.w_vector)
 
        elif self.loss_f == 'ce':
            def fun_error(target,output):
                if (self.filter_type == 'mio') | (self.filter_type == 'softmax'):
                    return cross_entropy(target = target,
                                                input = output,
                                                weight=self.w_vector.to(device=self.device),
                                                reduction='sum'
                                               ).to(self.device)
                    # return log_loss(target, output,normalize=False,eps=float)
                else:
                    raise Exception('BCE (Binary cross entropy) must work with filters either softmax or mio (max_is_one), {} was provided.'.format(self.filter_type))
        elif self.loss_f == 'mul_f1':
            def fun_error(target,output):
                return f1_score(target.cpu().numpy(),output.cpu().numpy(),average='macro')

        elif self.loss_f == 'mul_ap':
            def fun_error(target,output):

                # return average_precision_score(target.cpu().numpy(),output.cpu().numpy(),average='macro')
                return average_precision_score(target.cpu(),output.cpu(),average='macro')
        else:
            raise Exception('Loss function {} undefined, please available loss functions.'.format(self.loss_f))
        
        return fun_error

    
def mean_absolute_error(y, y_hat, w_vector):
    ''' Calculates the absoute error between the predicted
        and the target values with weigthed error per output
      Args:
        y     - (torch tensor) - target array tensor dimension [nSamples,1]
        y_hat - (torch tensor) - predicted array values dimension [nSamples,1]
        w_vector-(list floats) - list of weights for each output 

      Returns:
        error - (torch tensor) - absoute sum error between 
                                 predicted and target
    '''    
    error = 0
    for c in range(y.shape[1]):
        error += torch.abs(y[:,c]*w_vector[c] - y_hat[:,c]*w_vector[c]).sum()
    return error.mean()

    


    
# def calculate_prediction_error(y, y_hat):
#     ''' Calculates the absoute error between the predicted
#         and the target values
#       Args:
#         y     - (torch tensor) - target array tensor dimension [nSamples,1]
#         y_hat - (torch tensor) - predicted array values dimension [nSamples,1]

#       Returns:
#         error - (torch tensor) - absoute sum error between 
#                                  predicted and target
#     '''    
#     error = torch.abs(y.flatten() - y_hat.flatten()).sum()
#     return error



