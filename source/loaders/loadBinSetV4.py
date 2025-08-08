# -*- coding: utf-8 -*-
"""
Created on Feb 16, 2023

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
        self.target_label = ['normal','anomaly']

        self.set = pd.read_csv(filename)

    def get_df(self):
        return self.set
    
    def get_set (self,window_size):

        filter = (self.set['set_type'] == self.set_type)
        if filter.sum() == 0:
            raise Exception("set_type {} was not found on the dataset.".format(self.set_type))
        else:
            x = self.set.loc[filter,self.data_label].to_numpy()
            y = self.set.loc[filter,self.target_label].to_numpy()

            return x,y
                

