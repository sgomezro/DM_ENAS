import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
# from torcheval.metrics import BinaryF1Score, BinaryAUPRC

from sklearn.metrics import f1_score, average_precision_score


class bin_class():
    def __init__ (self,p,device,dtype=torch.float32):
        if p['nn_output_size'] > 2:
            raise OtherException('Type of class function selected is for binary classification, nevertheless, multi-class output size is selected.')
        # Setting cpu or gpu device
        self.device = device
        self.dtype  = dtype
        
        # Setting weights vector parameter
        if 'out_w_vector' in p:
            w_vector = p['out_w_vector']
        else:
            w_vector = np.ones(p['nn_output_size'])
        self.w_vector = torch.Tensor(w_vector)
        
        # Setting NN loss function
        self.loss_f = p['nn_loss_function']
        
        # Setting output filter
        if 'output_filter' in p:
            self.filter_type = p['output_filter']
        else:
            self.filter_type = None

    def loss_function(self):
        if self.loss_f == 'mae':
            def fun_error(target,output):
                return mean_absolute_error(target,output,self.w_vector)
 
        elif self.loss_f == 'bce':
            def fun_error(target,output):
                if (self.filter_type == 'mio') | (self.filter_type == 'softmax'):
                    return binary_cross_entropy(target.float(),
                                                output,
                                                weight=self.w_vector.to(device=self.device),
                                                reduction='sum'
                                               ).to(self.device)
                else:
                    raise Exception('BCE (Binary cross entropy) must work with filters either softmax or mio (max_is_one).')
        elif self.loss_f == 'bin_f1':
            # def fun_error(target,output):
            #     f1score = BinaryF1Score(device=self.device)
            #     f1score.update(output.flatten(),target.flatten())
            #     return f1score.compute()

            def fun_error(target,output):
                return f1_score(output.flatten().numpy(),target.flatten().numpy())

        elif self.loss_f == 'bin_ap':
            # # @torch.jit.trace
            # def fun_error(target,output):
            #     ap = BinaryAUPRC(device=self.device)
            #     ap.update(output.flatten(),target.flatten())
            #     return ap.compute()
            #     # return 1-ap.compute()
            
            def fun_error(target,output):
                return average_precision_score(output.flatten().numpy(),target.flatten().numpy())

        
        # elif self.loss_f == 'bin_hl':
        #     def fun_error(target,output):
        #         bhl = BinaryHammingDistance(device=self.device)
        #         return bhl(output,target)
        else:
            raise Exception('Loss function undefined, please available loss functions.')
        
        return fun_error
    
#     def loss_function(self):
#         if self.loss_f == 'mae':
#             def fun_error(target,output):
#                 return mean_absolute_error(target,output,self.w_vector)
 
#         elif self.loss_f == 'bce':
#             def fun_error(target,output):
#                 if (self.filter_type == 'mio') | (self.filter_type == 'softmax'):
#                     return binary_cross_entropy(target,output,
#                                                   weight=self.w_vector.to(device=self.device),
#                                                   reduction='sum')
#                 else:
#                     raise Exception('BCE (Binary cross entropy) must work with filters either softmax or mio (max_is_one).')
#         elif self.loss_f == 'bin_f1':
#             def fun_error(target,output):
#                 f1score = BinaryF1Score().to(self.device)
#                 return (1-f1score(output,target))
        
#         elif self.loss_f == 'bin_ap':
#             def fun_error(target,output):
#                 bap = BinaryAveragePrecision().to(self.device)
#                 # temp_target = target.to(torch.int32)
#                 return (1-bap(output,target.to(torch.int32)))
        
#         elif self.loss_f == 'bin_hl':
#             def fun_error(target,output):
#                 bhl = BinaryAveragePrecision().to(self.device)
#                 # temp_target = target.to(torch.int32)
#                 return bhl(output,target.to(torch.int32))
#         else:
#             raise Exception('Loss function undefined, please available loss functions.')
        
#         return fun_error


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







# def calculateMultiClassError(y, nn_output,w_vector,filter_type=None):
#     ''' Calculates clasification error using cross entropy for multiclass label
#       Args:
#         nn_output - (torch tensor) - predicted array values dimension [num samples,2]

#       Returns:
#         loss function result - (torch tensor) 
#     '''
    
#     loss =F.cross_entropy(nn_output,y.argmax(axis=1),weight=w_vector,reduction='sum')

#     return loss


