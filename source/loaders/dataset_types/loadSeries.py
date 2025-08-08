# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:05:57 2020

@author: Santiago
"""

import pandas as pd

import os
from os.path import dirname
cae_path = os.getcwd()

#df = pd.read_csv(path+"/h3_norm_multTargetsv1.csv")

class dataSet():
    def __init__(self,filename):
        self.df = pd.read_csv(cae_path+filename)


    def getDf(self):
        return df

    def getSet (self,setSplit=0.8):
        size = int(len(df)*(setSplit))
        dfTrain = self.df.iloc[:size,:]
        dfTest = self.df.iloc[size:,:]
        trainSet = dfTrain.drop(['Datetime','T+0','T+1','T+2','T+3','T+4', \
                     'T+5','T+6','T+7','House','SetType','isCMA' ], axis=1).to_numpy()
        trainTarget = dfTrain['T+0'].to_numpy()
        testSet = dfTest.drop(['Datetime','T+0','T+1','T+2','T+3','T+4', \
                     'T+5','T+6','T+7','House','SetType','isCMA'], axis=1).to_numpy()
        testTarget = dfTest['T+0'].to_numpy()
        return trainSet, trainTarget, testSet, testTarget

    def getDFSet(self,setSplit = 0.8):
        size = int(len(self.df)*(setSplit))
        dfTrain = self.df.iloc[:size,:]
        dfTest  = self.df.iloc[size:,:]
        trainSet= dfTrain.drop(['T+0','T+1','T+2','T+3','T+4', \
                     'T+5','T+6','T+7','House','SetType','isCMA'], axis=1)
        trainTarget = dfTrain['T+0']
        testSet = dfTest.drop(['T+0','T+1','T+2','T+3','T+4', \
                     'T+5','T+6','T+7','House','SetType','isCMA'], axis=1)
        testTarget = dfTest['T+0']
        return trainSet, trainTarget, testSet, testTarget

    def getLabels(self):
        labels = self.df.drop(['Datetime','T+0','T+1','T+2','T+3','T+4', \
                     'T+5','T+6','T+7','House','SetType','isCMA'], axis=1)
        labels = labels.columns
        return labels.to_list() + ['Load']

    def getDatatime (self):
        datetime = self.df['Datetime']
        return datetime.to_numpy()
