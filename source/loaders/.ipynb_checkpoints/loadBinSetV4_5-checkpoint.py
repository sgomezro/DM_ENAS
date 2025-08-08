# -*- coding: utf-8 -*-
"""
Created on Feb 16, 2023
updated on Jan 9, 2024

@author: Santiago
"""

import pandas as pd
import numpy as np


class data():
    def __init__(self,filename,input_dimension,set_type,col_target):
        self.filename = filename
        self.set_type = set_type
        #defining labels for X atributes and targets
        self.data_label = [str(i) for i in range(0,input_dimension)]
        self.target_label = col_target

    def get_df(self):
        return pd.read_csv(self.filename)
    
    def get_set (self):
        set = pd.read_csv(self.filename)
        mask = (set.set_type == self.set_type)
        if mask.sum() == 0:
            raise Exception("set_type {} was not found on the dataset.".format(self.set_type))
        else:
            indices = set.loc[mask,:].index.values
            inputs = set.loc[:,self.data_label].to_numpy()
            targets= set.loc[:,self.target_label].to_numpy()
            return inputs,targets,indices
                

