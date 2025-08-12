import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torchmetrics.functional.classification import binary_average_precision, binary_f1_score
from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error
from torchmetrics.classification import BinaryF1Score

class bin_class():
    def __init__ (self,p,device,dtype=None):
        if p['nn_output_size'] > 2:
            raise OtherException('Type of class function selected is for binary classification, nevertheless, multi-class output size is selected.')
        # Setting cpu or gpu device
        self.device = device
        self.dtype  = dtype

        # Setting NN loss function
        self.loss_f = p['nn_loss_function']
        
        # Setting output filter
        if 'output_filter' in p:
            self.filter_type = p['output_filter']
        else:
            self.filter_type = None

    def loss_function(self):
        if self.loss_f == 'mae':
            # mean_absolute_error = MeanAbsoluteError()
            def fun_error(target,preds):
                return mean_absolute_error(preds.contiguous(),target.contiguous()).to(self.device)

        elif self.loss_f == 'mse':
            # mean_square_error = MeanSquaredError(num_outputs=self.num_classes).to(self.device)
            def fun_error(target,preds):
                # print(f'target {target.size()}-{target.dtype}, predictions {preds.size()}-{preds.dtype}')
                # print(f'target {target[1,:]}, preds {preds[1,:]}')
                return mean_squared_error(preds.contiguous(),target.contiguous()).to(self.device)
 
        elif self.loss_f == 'bin_ce':
            def fun_error(target,preds):
                if (self.filter_type == 'mio') | (self.filter_type == 'softmax'):
                    return binary_cross_entropy(preds, target,reduction='sum').to(self.device)
                else:
                    raise Exception('bin_ce (binary cross entropy) must work with filters either softmax or mio (max_is_one), ->{} was provided.'.format(self.filter_type))
        
        elif self.loss_f == 'bin_f1':
            f1_score = BinaryF1Score().to(self.device)
            def fun_error(target,preds):
                return f1_score(preds.flatten(),target.flatten())

        elif self.loss_f == 'bin_ap':
            def fun_error(target,preds):
                return binary_average_precision(preds.flatten(), target.flatten().int()).to(self.device)
        else:
            raise Exception('Loss function {} undefined, please use available loss functions.'.format(self.loss_f))
        
        return fun_error
    




