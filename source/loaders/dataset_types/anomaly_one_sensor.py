# -*- coding: utf-8 -*-
"""
Created on Sept 21 7pm 2020

@author: Santiago
"""

import pandas as pd

import os,sys
from os.path import dirname
sys.path.append(str(os.getcwd()))
# appending storage 1 path to load directly from storage in the server
sys.path.append('/storage_1')


class dataset():
    def __init__(self,filename):
        self.df = pd.read_csv(filename)

        #defining labels for X atributes and targets
        self.data_label = ['reading']
        self.target_label = ['normal','missing','square','saturated','anomaly']

    def get_df(self):
        return self.df
    
    def get_ad_set (self,sensor):
        idx = self.df[(self.df['sensor'] == sensor) & (self.df['set_type'] == 'trainset')].index
        train_x = self.df.loc[idx,self.data_label].reset_index(drop=True).to_numpy()
        train_y = self.df.loc[idx,self.target_label].reset_index(drop=True).to_numpy()

        idx = self.df[(self.df['sensor'] == sensor) & (self.df['set_type'] == 'cma')].index
        cma_x = self.df.loc[idx,self.data_label].reset_index(drop=True).to_numpy()
        cma_y = self.df.loc[idx,self.target_label].reset_index(drop=True).to_numpy()

        idx = self.df[(self.df['sensor'] == sensor) & (self.df['set_type'] == 'testset')].index
        test_x = self.df.loc[idx,self.data_label].reset_index(drop=True).to_numpy()
        test_y = self.df.loc[idx,self.target_label].reset_index(drop=True).to_numpy()

        setDict = {'train_x':train_x,
                   'train_y':train_y,
                   'cma_x'  :cma_x,
                   'cma_y'  :cma_y,
                   'test_x' :test_x,
                   'test_y' :test_y} 
        return setDict


#     def getDFMHSet(self):
#         dLtemp = self.d_label.copy()
#         dLtemp.remove('Datetime')
#         dLtemp.remove('House')
#         df_a = self.df[self.df['SetType'] == 'Train']
#         df_t = df_a.drop(dLtemp, axis=1)
#         trainXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
#         trainYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]

#         df_a = self.df[self.df['SetType'] == 'Test']
#         df_t = df_a.drop(dLtemp, axis=1)
#         testXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
#         testYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]

#         df_a = self.df[self.df['isCMA']]
#         df_t = df_a.drop(dLtemp, axis=1)
#         cmaXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
#         cmaYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]
#         setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
#         return setList

#     def getLabels(self):
#         labels = self.df.drop(self.d_label, axis=1)
#         labels = labels.columns
#         return labels.to_list()

#     def getDatatime (self):
#         datetime = self.df['Datetime']
#         return datetime.to_numpy()

#     def getDFMHMSSet(self,targ_size=8):
#         dLtemp = self.d_label.copy()
#         dLtemp.remove('Datetime')
#         dLtemp.remove('House')
#         df_a = self.df[self.df['SetType'] == 'Train']
#         df_t = df_a.drop(dLtemp, axis=1)
#         trainXL = df_t.drop(['House'], axis=1).reset_index(drop=True) 
#         trainYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)

#         df_a = self.df[self.df['SetType'] == 'Test']
#         df_t = df_a.drop(dLtemp, axis=1)
#         testXL = df_t.drop(['House'], axis=1).reset_index(drop=True) 
#         testYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)

#         df_a = self.df[self.df['isCMA']]
#         df_t = df_a.drop(dLtemp, axis=1)
#         cmaXL = df_t.drop(['House'], axis=1).reset_index(drop=True)
#         cmaYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)
#         setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
#         return setList

#     def getMHMSSet (self,targ_size=8):
#         dLtemp = self.d_label.copy()
#         dLtemp.remove('House')
#         df_a = self.df[self.df['SetType'] == 'Train']
#         df_t = df_a.drop(dLtemp, axis=1)
#         trainXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
#         trainYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]


#         df_a = self.df[self.df['SetType'] == 'Test']
#         df_t = df_a.drop(dLtemp, axis=1)
#         testXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
#         testYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]

#         df_a = self.df[self.df['isCMA']]
#         df_t = df_a.drop(dLtemp, axis=1)
#         cmaXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
#         cmaYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]
#         setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
#         return setList

