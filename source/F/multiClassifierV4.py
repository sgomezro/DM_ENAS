import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torchmetrics.classification import MulticlassF1Score, MulticlassAveragePrecision, MultilabelF1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.functional.classification import average_precision, f1_score
from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error

class multi_class():
    def __init__ (self,p,device,dtype=None):
        if p['nn_output_size'] < 2:
            raise OtherException('Type of class function selected is for multi classification, nevertheless, binary-class output size is selected.')
        self.num_classes = p['nn_output_size']
        # Setting cpu or gpu device
        self.device = torch.cpu
        self.dtype  = dtype

        # Setting NN loss function
        self.loss_f = p['nn_loss_function']
        
        # # Setting weights vector parameter
        # if 'out_w_vector' in p:
        #     if ('mae' in self.loss_f) | ('ce' in self.loss_f):
        #         w_vector = p['out_w_vector']
        #     else:
        #         raise Exception('weigth in multi classes supported only for mae and ce loss functions.')
        #         # w_vector = np.ones(p['nn_output_size'])
        # else:
        #     w_vector = np.ones(p['nn_output_size'])
        # self.w_vector = torch.Tensor(w_vector).to(device=self.device)
        

        
        # Setting output filter
        if 'output_filter' in p:
            self.filter_type = p['output_filter']
        else:
            self.filter_type = None

    def loss_function(self):
        if self.loss_f == 'mae':
            # mean_absolute_error = MeanAbsoluteError()
            def fun_error(target,preds):
                return mean_absolute_error(preds.contiguous(),target.contiguous())

        elif self.loss_f == 'mse':
            # mean_square_error = MeanSquaredError(num_outputs=self.num_classes).to(self.device)
            def fun_error(target,preds):
                # print(f'target {target.size()}-{target.dtype}, predictions {preds.size()}-{preds.dtype}')
                # print(f'target {target[1,:]}, preds {preds[1,:]}')
                return mean_squared_error(preds.contiguous(),target.contiguous(),num_outputs=self.num_classes)
 
        elif self.loss_f == 'mul_ce':
            def fun_error(target,preds):
                if (self.filter_type == 'mio') | (self.filter_type == 'softmax'):
                    return cross_entropy(preds, target,reduction='sum')
                                         # weight=self.w_vector,
                                         # reduction='sum')
                else:
                    raise Exception('mul_ce (multiple cross entropy) must work with filters either softmax or mio (max_is_one), ->{} was provided.'.format(self.filter_type))
        
        elif self.loss_f == 'mul_f1':
            # f1_score = MulticlassF1Score(num_classes=self.num_classes,average='macro')
            def fun_error(target,preds):
                return f1_score(preds,target.argmax(axis=1), task="multiclass", num_classes=self.num_classes,average='macro')

        elif self.loss_f == 'mlab_f1':
            # f1_score =  MultilabelF1Score(num_labels=self.num_classes,average='macro')
            def fun_error(target,preds):
                return f1_score(preds,target, task='multilabel', num_labels=self.num_classes,average='macro')

        elif self.loss_f == 'mul_ap':
            # ap_score = MulticlassAveragePrecision(num_classes=self.num_classes,average='macro')
            def fun_error(target,preds):
                # return ap_score(preds,target.argmax(axis=1))
                return average_precision(preds, target.argmax(axis=1), task='multiclass', num_classes=self.num_classes, average='macro')
        else:
            raise Exception('Loss function {} undefined, please use available loss functions.'.format(self.loss_f))
        
        return fun_error
    
# def mean_absolute_error(y_hat, y, w_vector):
#     ''' Calculates the absoute error between the predicted
#         and the target values with weigthed error per output
#       Args:
#         y     - (torch tensor) - target array tensor dimension [nSamples,1]
#         y_hat - (torch tensor) - predicted array values dimension [nSamples,1]
#         w_vector-(list floats) - list of weights for each output 

#       Returns:
#         error - (torch tensor) - absoute sum error between 
#                                  predicted and target
#     '''    
#     error = 0
#     for c in range(y.shape[1]):
#         error += torch.abs(y[:,c]*w_vector[c] - y_hat[:,c]*w_vector[c]).sum()
#     return error.mean()

# def mean_square_error(y_hat, y, w_vector):
#     ''' Calculates the absoute error between the predicted
#         and the target values with weigthed error per output
#       Args:
#         y     - (torch tensor) - target array tensor dimension [nSamples,1]
#         y_hat - (torch tensor) - predicted array values dimension [nSamples,1]
#         w_vector-(list floats) - list of weights for each output 

#       Returns:
#         error - (torch tensor) - absoute sum error between 
#                                  predicted and target
#     '''    
#     error = 0
#     for c in range(y.shape[1]):
#         error += ((y[:,c]*w_vector[c] - y_hat[:,c]*w_vector[c])**2).sum()
#     return error.mean()
    


    
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



