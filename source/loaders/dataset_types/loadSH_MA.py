# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 23:05:57 2020

@author: Santiago
"""

import pandas as pd

import os
from os.path import dirname
cae_path = os.getcwd()

class dataSet():
    def __init__(self,filename):
        self.df = pd.read_csv(cae_path+filename)

        #defining labels for X atributes and targets
        self.d_label = ['Datetime','SA0','SA1','SA2','SA3','SA4', \
                         'SA5','SA6','SA7','SetType','isCMA','House']
        self.tg_label = ['SA0','SA1','SA2','SA3','SA4', 'SA5','SA6','SA7']

    def getDf(self):
        return self.df

    def getMHSet (self):
        dLtemp = self.d_label.copy()

        dLtemp.remove('House')
        df_a = self.df[self.df['SetType'] == 'Train']
        df_t = df_a.drop(dLtemp, axis=1)
        trainXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        trainYL = [df_a[self.tg_label[0]].reset_index(drop=True).to_numpy()]


        df_a = self.df[self.df['SetType'] == 'Test']
        df_t = df_a.drop(dLtemp, axis=1)
        testXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        testYL = [df_a.loc[:,self.tg_label[0]].reset_index(drop=True).to_numpy()]

        df_a = self.df[self.df['isCMA']]
        df_t = df_a.drop(dLtemp, axis=1)
        cmaXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        cmaYL = [df_a.loc[:,self.tg_label[0]].reset_index(drop=True).to_numpy()]
        setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
        return setList


    def getDFMHSet(self):
        dLtemp = self.d_label.copy()
        dLtemp.remove('Datetime')
        dLtemp.remove('House')
        df_a = self.df[self.df['SetType'] == 'Train']
        df_t = df_a.drop(dLtemp, axis=1)
        trainXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
        trainYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]

        df_a = self.df[self.df['SetType'] == 'Test']
        df_t = df_a.drop(dLtemp, axis=1)
        testXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
        testYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]

        df_a = self.df[self.df['isCMA']]
        df_t = df_a.drop(dLtemp, axis=1)
        cmaXL = [df_t[df_t['House'] == i].drop(['House'], axis=1).reset_index(drop=True) for i in self.df['House'].unique()]
        cmaYL = [df_a[df_a['House'] == i][self.tg_label[0]].reset_index(drop=True) for i in self.df['House'].unique()]
        setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
        return setList

    def getLabels(self):
        labels = self.df.drop(self.d_label, axis=1)
        labels = labels.columns
        return labels.to_list()

    def getDatatime (self):
        datetime = self.df['Datetime']
        return datetime.to_numpy()

    def getDFMHMSSet(self,targ_size=8):
        dLtemp = self.d_label.copy()
        dLtemp.remove('Datetime')
        dLtemp.remove('House')
        df_a = self.df[self.df['SetType'] == 'Train']
        df_t = df_a.drop(dLtemp, axis=1)
        trainXL = df_t.drop(['House'], axis=1).reset_index(drop=True) 
        trainYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)

        df_a = self.df[self.df['SetType'] == 'Test']
        df_t = df_a.drop(dLtemp, axis=1)
        testXL = df_t.drop(['House'], axis=1).reset_index(drop=True) 
        testYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)

        df_a = self.df[self.df['isCMA']]
        df_t = df_a.drop(dLtemp, axis=1)
        cmaXL = df_t.drop(['House'], axis=1).reset_index(drop=True)
        cmaYL = df_a[self.tg_label[:targ_size]].reset_index(drop=True)
        setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
        return setList

    def getMHMSSet (self,targ_size=8):
        dLtemp = self.d_label.copy()
        dLtemp.remove('House')
        df_a = self.df[self.df['SetType'] == 'Train']
        df_t = df_a.drop(dLtemp, axis=1)
        trainXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        trainYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]


        df_a = self.df[self.df['SetType'] == 'Test']
        df_t = df_a.drop(dLtemp, axis=1)
        testXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        testYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]

        df_a = self.df[self.df['isCMA']]
        df_t = df_a.drop(dLtemp, axis=1)
        cmaXL = [df_t.drop(['House'], axis=1).reset_index(drop=True).to_numpy()]
        cmaYL = [df_a[self.tg_label[:targ_size]].reset_index(drop=True).to_numpy()]
        setList = [trainXL, trainYL, testXL, testYL, cmaXL, cmaYL]
        return setList

