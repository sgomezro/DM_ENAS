# -*- coding: utf-8 -*-
"""
Created on Oct 11, 2023

@author: Santiago
"""

import pandas as pd
import numpy as np


class data():
    def __init__(self,filename,set_type):
        self.filename = filename
        self.set_type = set_type
        #defining labels for X atributes and targets
        self.data_label = ['reading']
        self.target_label = ['normal','missing','square','saturated','anomaly']

        self.set = pd.read_csv(filename)

    def get_df(self):
        return self.set
    
    def get_set (self,window_size):
        
        filter = (self.set['set_type'] == self.set_type)
        if filter.sum() == 0:
            raise Exception("set_type {} was not found on the dataset.".format(self.set_type))
        else:
            if self.set_type == 'trainset':
                x = self.set.loc[filter,self.data_label].to_numpy()
                y = self.set.loc[filter,self.target_label].to_numpy()
            else:
                idx = self.set[filter].index
                x = np.empty((0,window_size), float)
                y = np.empty((0,len(self.target_label)),np.int32)
    
                for ix in idx:
                    if ix > window_size:
                        x = np.append(x,self.set.loc[ix-window_size+1:ix,self.data_label].\
                                      to_numpy().T,axis=0)
                        y = np.append(y,self.set.loc[ix,self.target_label].\
                                      to_numpy(np.int32).reshape(1,len(self.target_label)),axis=0)

            return x,y
                



