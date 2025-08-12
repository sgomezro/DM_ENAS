import numpy as np
import torch
import copy

class loader():
    def __init__ (self,p,set_type,dtype):
        self.nn_input = p['nn_input_size']
        
        self.start = 0          # Current starting point in the training set
        self.batch = p['batch_size']
        self.missing_batch = int(p['batch_size']*0.01)
        self.set_type = set_type
        self.dtype = dtype
        self.task = p['task']
        self.input_dimension = p['nn_input_size']
        if p['window_size'] > 0:
            self.window_size = p['window_size']

        self.inputs = []
        self.targets= []
        self.set_type_idx = []
        # self.train_x = []
        # self.train_y = []
        # self.test_x  = []
        # self.test_y  = []
        # self.adj_w_x   = []
        # self.adj_w_y   = []
        # self.out_label = []

        # if 'target_size' in p:
        #     # getting columns from target according to p['out_target']
        #     labels = data.target_label
        #     self.col_target = [labels.index(i) for i in p['target_size']]
        #     p['nn_output_size'] = len(self.col_target)
        # else:
        #     self.col_target = np.arange(p['nn_output_size'])         
        self.col_target = p['target_size']
        data = self.set_from_task(p['data_path'],set_type)
        self.init_dataset(data)
        # del data    

    def update_window_size(self,w_size):
        # self.nn_inpu = inputs_size
        self.window_size = w_size
        
    def set_from_task(self,datapath,set_type):
        # selecting the dataset for the selected task 
        if self.task == 'House': #for house forecasting
            print('Need to implement wrapper in loader')
            
        # elif self.task == 'AD_singleSensor': #for anomaly detection big dataset
        #     from .dataset_types.anomaly_one_sensor import dataset
        elif self.task == 'ad_shms': #for anomly detection whith structural healt monitoring systems (bridge data).
            from .loadShmsSetV4 import data

        #for anomly detection with msl or smap or yahoo datasets
        elif any(self.task == ad_set for ad_set in ['ad_msl','ad_smap','ad_yahoo']): 
            from .loadBinSetV4_5 import data
           
        else:
            raise Exception("Must define a type of task: House, AD_single_sensor or ad_smap in parameters set")
        return data(datapath,self.input_dimension,set_type,self.col_target)    
        
    def init_dataset(self,data):
        # returning anomaly set depending of the task
        self.inputs,self.targets,self.set_type_idx = data.get_set()

            
        # if  self.set_type == 'trainset':
        #     self.train_x = copy.deepcopy(x)
        #     self.train_y = copy.deepcopy(y[:,self.col_target])
        # elif self.set_type == 'testset':
        #     self.test_x  = copy.deepcopy(x)
        #     self.test_y  = copy.deepcopy(y[:,self.col_target])
        # elif self.set_type == 'adj_w':
        #     self.adj_w_x   = copy.deepcopy(x)
        #     self.adj_w_y   = copy.deepcopy(y[:,self.col_target])
        # else:
        #     raise Exception("Must select a set type between trainset, adj_w_set, or testset")
        del data
    
    def adj_weight_set(self): #adjust weight set
        if len(self.set_type_idx) < 1:
            raise Exception("Must init adj_w set")
        else:
            x,y = self.sliding_window()
            
            return x, y
        
    def testset (self):
        if len(self.set_type_idx) < 1:
            raise Exception("Must init testset")
        else:
            x,y = self.sliding_window()

            return x, y
    
    def get_train_start_idx(self,rng):
        if (len(self.set_type_idx)-self.batch) <= 0:
            raise Exception("batch size higher than trainset size! Correct batch size")
        else:
            start = 0
            if self.batch > 0:  # actions when a batch is defined to retrieve
                            # a portion of the training set
                start = rng.integers(self.window_size+self.batch,high=self.train_x.shape[0]) 
            return start
            
    def trainset_from_start_idx(self,start):
        if len(self.set_type_idx) < 1:
            raise Exception("Must init train dataset")
        elif (len(self.set_type_idx)-self.batch) < 1:
            raise Exception("batch size higher than trainset size! Correct batch size")
        else:
            if self.batch > 0:  # actions when a batch is defined to retrieve a portion of the training set
                print('Needs implementation')
                # # curr_idx = [i for i in range(start,start+self.batch)]
                # # return self.train_x[curr_idx], self.train_y[curr_idx]
                # input = self.train_x[start-self.window_size+1:start+self.batch]
                # target= self.train_y[start-self.window_size+1:start+self.batch]
                
                # if np.any(target.sum(axis=0) == 0):
                #     missing_classes = np.where(target.sum(axis=0) == 0)[0]
                #     x_add,y_add = self.adding_missing_classes(missing_classes,self.train_x,self.train_y)
                #     input = np.concatenate((input,x_add),axis=0)
                #     target= np.concatenate((target,y_add),axis=0)
                    
                # x,y = self.sliding_window(input,target)

            else:
                x,y = self.sliding_window()

            return x, y

    def sliding_window(self):
        """
        Create an improved sliding window function for a 2D array.
    
        Each input sample is transformed into a 1D array, arranged such that
        the elements go through each column of the window.
    
        :self.inputs: 2D array of shape (samples, sensors)
        :self.target: 1D array of labels
        :self.self.set_type_idx: List of self.set_type_idx to create windows from
        :self.window_size: Size of the sliding window
        :return: Tuple of two lists containing windows for inputs and target
        """

        valid_indices = [index for index in self.set_type_idx if index >= self.window_size - 1]
        n_samples = len(valid_indices)
        sensor_size = self.inputs.shape[1]
        
        x_slide = torch.zeros((n_samples,sensor_size*self.window_size)).to(self.dtype)
        if np.ndim(self.targets) == 1:
            y_slide = torch.empty(n_samples).to(self.dtype)
        else:
            y_slide = torch.empty(n_samples,self.targets.shape[1]).to(self.dtype)
    
        for i,index in enumerate(valid_indices):
            # Initialize empty window
            input_window = []
    
            # Populate the window by column
            for sensor_idx in range(self.inputs.shape[1]):
                column = []
                for window_idx in range(self.window_size):
                    column.append(self.inputs[index - window_idx, sensor_idx])
                input_window.extend(column)
    
            # target_window = target[index]
    
            x_slide[i] = torch.tensor(input_window).to(self.dtype)
            y_slide[i] = torch.tensor(self.targets[index]).to(self.dtype)
    
        return x_slide, y_slide
    
    def adding_missing_classes (self,missing_classes,x,y):
        num_samples = int(self.missing_batch/self.window_size)
        y_add = np.empty((len(missing_classes),num_samples,\
                          self.window_size,y.shape[1]),y.dtype)
        x_add = np.empty((len(missing_classes),num_samples,\
                          self.window_size,x.shape[1]),x.dtype)
        for c,miss_c in enumerate(missing_classes):
            idx = np.where(y[:,miss_c] > 0)[0]
            idx0 = np.where(idx > self.window_size)[0][:num_samples]
            idx = idx[idx0]
            # print(f'class {miss_c},  idx0 {idx0} \n idx {idx}')

            for i,ix in enumerate(idx):
                y_add[c,i] = y[ix-self.window_size+1:ix+1,:]
                x_add[c,i] = x[ix-self.window_size+1:ix+1,:]
        y_add = y_add.reshape(-1,y.shape[-1])
        x_add = x_add.reshape(-1,x.shape[-1])
        print(f'Classes {missing_classes} missing targets. Adding {num_samples} samples with window lenght {self.window_size} to the batch. In total {y_add.shape[0]} entries added.')
        return x_add, y_add


#### Deprecated#####        
    # def sliding_window(self,input,target):
    #     ''' returns a set with the sliding window given a sequence
    #     args:
    #         input: np array of size size [C+window size]x1 with input data in sequence
    #         target: np array of size [C+window size]xtarget clases with target clases data in sequence
    #     output:
    #         x: np array: Cx(window size) set with sliding window for sequence
    #         y: np array: Cx(target clases) set with target clases'''
        
    #     size = len(input)-self.window_size+1
    #     x_slide = torch.zeros((size,self.window_size),dtype=self.dtype)
    #     input_tensor = torch.from_numpy(input).to(self.dtype)
    #     for i in range(self.window_size):
    #         #Rotate sequence and add columns instead of rows
    #         x_slide[:,i] = input_tensor[i:i+size].flatten()
    #     y_slide = torch.from_numpy(target[self.window_size-1:,:]).to(self.dtype)
        
    #     return x_slide, y_slide
        