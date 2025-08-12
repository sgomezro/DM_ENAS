import numpy as np
import copy

class loader():
    def __init__ (self,p,set_type):
        self.nn_input = p['nn_input_size']
        
        self.start = 0          # Current starting point in the training set
        self.batch = p['batch_size']
        self.set_type = set_type
        self.task = p['task']
        if p['window_size'] > 0:
            self.window_size = p['window_size']
        
        self.train_x = []
        self.train_y = []
        self.test_x  = []
        self.test_y  = []
        self.adj_w_x   = []
        self.adj_w_y   = []
        self.out_label = []
        
        data = self.set_from_task(p['data_path'],set_type)
        
        if 'target_size' in p:
            # getting columns from target according to p['out_target']
            labels = data.target_label
            self.col_target = [labels.index(i) for i in p['target_size']]
            p['nn_output_size'] = len(self.col_target)
        else:
            self.col_target = np.arange(p['nn_output_size']) 
        
        self.init_dataset(data)
        # del data        
        
    def set_from_task(self,datapath,set_type):
        # selecting the dataset for the selected task 
        if self.task == 'House': #for house forecasting
            print('Need to implement wrapper in loader')
            
        # elif self.task == 'AD_singleSensor': #for anomaly detection big dataset
        #     from .dataset_types.anomaly_one_sensor import dataset
        elif self.task == 'ad_shms': #for anomly detection whith structural healt monitoring systems (bridge data).
            from .load_shms_setV3_3 import data
        
        elif self.task == 'ad_smap': #for anomly detection smap dataset
            from .load_smap_set import data
            
        elif self.task == 'ad_msl': #for anomly detection smap dataset
            from .load_msl_set import data
        
        elif self.task == 'ad_yahoo': #for anomly detection smap dataset
            from .load_yahoo_set import data
            
        else:
            raise Exception("Must define a type of task: House, AD_single_sensor or ad_smap in parameters set")
        return data(datapath,set_type)    
        
    def init_dataset(self,data):
        # returning anomaly set depending of the task
        x,y = data.get_set(self.window_size)
            
        if  self.set_type == 'trainset':
            self.train_x = copy.deepcopy(x)
            self.train_y = copy.deepcopy(y[:,self.col_target])
        elif self.set_type == 'testset':
            self.test_x  = copy.deepcopy(x)
            self.test_y  = copy.deepcopy(y[:,self.col_target])
        elif self.set_type == 'adj_w':
            self.adj_w_x   = copy.deepcopy(x)
            self.adj_w_y   = copy.deepcopy(y[:,self.col_target])
        else:
            raise Exception("Must select a set type between trainset, adj_w_set, or testset")
        del data
        print('reached upto init_dataset and finished')
    
    def adj_weight_set(self): #adjust weight set
        if len(self.adj_w_x) <= 0:
            raise Exception("Must init adj_w set")
        else:
            return self.adj_w_x, self.adj_w_y
        
    def testset (self):
        if len(self.test_x) <= 0:
            raise Exception("Must init testset")
        else:
            return self.test_x, self.test_y
    
    def get_train_start_idx(self,rng):
        if len(self.train_x) <= 0:
            raise Exception("Must init train dataset")
        elif (self.train_x.shape[0]-self.batch) <= 0:
            raise Exception("batch size higher than trainset size! Correct batch size")
        else:
            start = 0
            if self.batch > 0:  # actions when a batch is defined to retrieve
                            # a portion of the training set
                start = rng.integers(self.nn_input,high=self.train_x.shape[0]-self.batch) 
            return start
            
    def trainset_from_start_idx(self,start):
        if len(self.train_x) <= 0:
            raise Exception("Must init train dataset")
        elif (self.train_x.shape[0]-self.batch) <= 0:
            raise Exception("batch size higher than trainset size! Correct batch size")
        else:
            if self.batch > 0:  # actions when a batch is defined to retrieve a portion of the training set
                # curr_idx = [i for i in range(start,start+self.batch)]
                # return self.train_x[curr_idx], self.train_y[curr_idx]
                curr_idx = np.arange(start,start+self.batch)
                x = np.empty((self.batch,self.window_size), float)
                # y = np.empty((self.batch,len(self.col_target)),np.int32)
                for i,idx in enumerate(curr_idx):
                    x[i,:] = self.train_x[idx-self.window_size:idx].flatten()
                y = self.train_y[curr_idx]
                return x,y
            else:
                return self.train_x, self.train_y

            