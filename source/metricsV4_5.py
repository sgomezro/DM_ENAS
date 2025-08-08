from sklearn.metrics import recall_score, precision_score, f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score
# from torchmetrics.functional.classification import binary_average_precision, binary_f1_score,BinaryAUROC
from torchmetrics.functional import auroc,average_precision, precision, recall, f1_score

import torch
import numpy as np
import pandas as pd
import zipfile
import os

class metric_class():
    def __init__ (self,p,device,average='macro'):
        self.device = device
        if ('target_result' in p) & ('target_size' in p):
            self.target_col = [p['target_size'].index(i) for i in p['target_result']]
            self.labels = [p['target_size'][i] for i in self.target_col]

        elif 'target_size' in p:
            self.labels = p['target_size']
            self.target_col = np.arange(len(p['target_size']))
        self.task_type = p['task_type']
        if self.task_type == 'binary_classification':
            self.average = 'bin'
            if len(self.labels) > 2:
                raise Exception(f'Binary classification was selected, but number of label is higher than 2.')
        elif self.task_type == 'multi_classification':
            self.average = average

    def set_class_values(self,target,preds):
        if self.task_type == 'binary_classification':
            self.target = target[:,self.target_col].to(self.device).flatten().int()
            self.predict= preds.to(self.device).flatten()
            # self.prob_predict = self.predict[:,self.target_col]
            # self.labels = self.target_labels
            
        elif self.task_type =='multi_classification':
            self.target = torch.tensor(target[:,self.target_col])
            self.predict= torch.tensor(preds)
            # self.prob_predict = self.predict[:,self.target_col]
            # self.labels = self.target_labels
            # self.init_values()
        else:
            raise Exception(f'Either binary_classification or multiclass task type are defined. {self.task_type} was provided')


    def get_score_report(self):
        if self.task_type == 'binary_classification':
            return self.get_bin_score_report()
        elif self.task_type =='multi_classification':
            return self.get_multi_score_report()
        else:
            raise Exception('Either binary_classification or multiclass task type are defined. {} was given'.format(self.task_type))
            
    def get_bin_score_report(self): #needs revision
        '''
        calculate different metrics for multi binary scores
        '''
        scores =  np.array([auroc(self.predict,self.target,task='binary').cpu(),
                    average_precision(self.predict,self.target,task='binary').cpu(),
                    precision(self.predict,self.target,task='binary').cpu(),
                    recall(self.predict,self.target,task='binary').cpu(),
                    f1_score(self.predict,self.target,task='binary').cpu() ])
        
        return scores

    def ensamble_score_report(self,scores,ensamble_from):
        if self.task_type == 'binary_classification':
            headers = [m+self.average for m in ['AUROC_','AP_','precision_', 'recall_','f1-score_']]
        elif self.task_type =='multi_classification':
            headers = [m+self.average for m in ['AP_','precision_', 'recall_','f1-score_']]
        
        if ensamble_from == 'scores_average':
            scores_avg = np.mean(scores,axis=0)
            return  pd.DataFrame([scores_avg],columns=headers)
        elif ensamble_from == 'confusion_matrix':
            aggregated_cm = sum(cm for cm,ap in scores)
            avg_ap = np.mean([ap for cm,ap in scores])
            precision,recall,f1 = f1_score_from_cm(aggregated_cm,average=self.average)
            return  pd.DataFrame([[avg_ap,precision,recall,f1]],columns=headers)
        elif ensamble_from == 'single_report':
            return pd.DataFrame([scores],columns=headers)
        else:
            raise Exception('Ensamble df score only from score_average or confusion_matrix. {} was given'.format(ensamble_from))
        
    # def set_predict(self,fitness):
    #     self.y = fitness.y.numpy()
    #     self.y_hat = fitness.y_hat.numpy()
        
    # def init_values(self):
    #     self.cat_target = self.target.astype(int).argmax(axis=1)
    #     self.cat_predict= np.argmax(self.predict,axis=1)
        
    #     #checking if target and predcit have the same unique values, otherwise adds the missing unique
    #     target_unique = np.unique(self.cat_target)
    #     labels_unique = [i for i in range(len(self.labels))]
    #     missing_unique = set(labels_unique).difference(set(target_unique))
    #     if len(missing_unique) > 0:
    #         for e in missing_unique:
    #             self.cat_target = np.insert(self.cat_target,0,e)
    #             self.cat_predict= np.insert(self.cat_predict,0,e)
            
    # def set_from_csv(self,filename,n_rows=None):
    #     print('Loading {}'.format(filename))
    #     if n_rows == None:
    #         df = pd.read_csv(filename,index_col=0)
    #     else:
    #         df = pd.read_csv(filename,index_col=0,nrows=n_rows)

    #     target_col = ['target_'+t for t in self.target_labels]
    #     predict_col = ['predict_'+t for t in self.target_labels]
    #     self.target = df.loc[:,target_col].to_numpy()
    #     self.predict= df.loc[:,predict_col].to_numpy()
    #     # self.prob_predict = torch.softmax(torch.Tensor(self.predict),dim=1).numpy()
    #     self.prob_predict = self.predict
    #     self.labels = self.target_labels
    #     if n_rows != None:
    #         # setting at least 1 for each class, for testing purpouses

    #         n_clases = len(target_col)
    #         for i in range(n_clases):
    #             uniq = np.unique(self.target[:,i])
    #             if len(uniq) < 2:
    #                 self.target[i,i] = abs(uniq[0]-1)
    #             uniq = np.unique(self.predict[:,i])
    #             if len(uniq) < 2:
    #                 self.predict[i,i] = abs(uniq[0]-1)
    #     self.init_values()

    def set_from_zip(self,filename,file_temp,zip_fname,n_rows=None):
        print('Loading {}'.format(filename))
        with zipfile.ZipFile(zip_fname, 'r') as zip_ref:
            zip_ref.extract(filename,file_temp)

        if n_rows == None:
            df = pd.read_pickle(file_temp+filename,compression='gzip')
        else:
            df = pd.read_pickle(file_temp+filename,nrows=n_rows,compression='gzip')
        os.remove(file_temp+filename)

        target_col  = ['target_'+t for t in self.target_labels]
        predict_col = ['predict_'+t for t in self.target_labels]
        target = df.loc[:,target_col].to_numpy()
        preds  = df.loc[:,predict_col].to_numpy()
        if self.task_type == 'binary_classification':
            self.target = target.flatten() 
            self.predict= preds.flatten()
        elif self.task_type =='multi_classification':
            self.target = target
            self.predict= preds
        else:
            raise Exception(f'Either binary_classification or multiclass task type are defined. {self.task_type} was provided')
        # self.prob_predict = self.predict
        # self.labels = self.target_labels
        
        if n_rows != None:
            # setting at least 1 for each class, for testing purpouses
            n_clases = len(target_col)
            for i in range(n_clases):
                uniq = np.unique(self.target[:,i])
                if len(uniq) < 2:
                    self.target[i,i] = abs(uniq[0]-1)
                uniq = np.unique(self.predict[:,i])
                if len(uniq) < 2:
                    self.predict[i,i] = abs(uniq[0]-1)
        # self.init_values()
        

    

        

    
    def get_multi_score_report(self,score_get):
        '''
        calculate different metrics for multi classification scores
        '''
        
        if score_get == 'full':
            report = classification_report(self.cat_target,
                                           self.cat_predict,
                                           target_names=self.labels,
                                           output_dict=True,zero_division=0)
            df = pd.DataFrame.from_dict(report)
        
        elif score_get == 'one_row': # getting scores for csv in one row
            # confirming classes exist for each column
            for c in range(len(self.target_labels)):
                if len(np.unique(self.target[:,c])) > 2:
                    raise ValueError('Only binary classes are allowed for ROC score')
                elif len(np.unique(self.target[:,c])) != 2:
                    self.target  = np.delete(self.target,c,1)
                    self.predict = np.delete(self.predict,c,1)
                    self.prob_predict= np.delete(self.prob_predict,c,1)
            # generating the dataframe with scores and labels
            col = [m+self.average for m in ['AP_','precision_', 'recall_','f1-score_']]
            aux =  [average_precision_score(self.target,self.prob_predict,average=self.average),
                    precision_score(self.target,self.prob_predict,average=self.average),
                    recall_score(self.target,self.prob_predict,average=self.average),
                    f1_score(self.target,self.prob_predict,average=self.average) ]
            df = pd.DataFrame([aux],columns=col)
        
        else:
            df = pd.DataFrame.from_dict(report).loc[score_get]
        return df





        
    def get_results(self):
        cats = self.labels
        pred_label = ['predict_'+ cat for cat in cats]
        target_label = ['target_'+ cat for cat in cats]
        # predict = pd.DataFrame(self.predict,columns=pred_label)
        predict = pd.DataFrame(self.prob_predict,columns=pred_label)
        target  = pd.DataFrame(self.target,columns=target_label)

        return pd.concat([target,predict],axis=1)



        

    
            
    def get_confusion_matrix(self):
        if self.task_type == 'binary_classification':
            return self.bin_confusion_matrix()
        elif self.task_type =='multi_classification':
            return self.multi_confusion_matrix()
        else:
            raise Exception('Either binary_classification or multiclass task type are defined. {} was given'.format(self.task_type))

    def bin_confusion_matrix(self):
        print(f'raw target {self.target}')
        print(f'predict {self.prob_predict}')
        cm = confusion_matrix(self.target.argmax(axis=1),self.prob_predict.argmax(axis=1))
        ap = average_precision_score(self.target.flatten(),self.prob_predict.flatten(),average=self.average)
        return error

    def multi_confusion_matrix(self):
        # for c in range(len(self.target_labels)):
        #     if len(np.unique(self.target[:,c])) > 2:
        #         raise ValueError('Only binary classes are allowed for ROC score')
        #     elif len(np.unique(self.target[:,c])) != 2:
        #         self.target  = np.delete(self.target,c,1)
        #         self.predict = np.delete(self.predict,c,1)
        #         self.prob_predict= np.delete(self.prob_predict,c,1)
        
        # generating the dataframe with scores and labels
        n_classes = len(self.labels)
        cm = confusion_matrix(self.target.argmax(axis=1),
                              self.prob_predict.argmax(axis=1),
                             labels=range(n_classes))
        ap = average_precision_score(self.target,self.prob_predict,average=self.average)
        # scores =  [average_precision_score(self.target,self.prob_predict,average=self.average),
        #         precision_score(self.target,self.prob_predict,average=self.average),
        #         recall_score(self.target,self.prob_predict,average=self.average),
        #         f1_score(self.target,self.prob_predict,average=self.average) ]
        return cm,ap


def f1_score_from_cm(cm, average='macro'):
    """Computes F1 scores given the aggragated confusion matrix"""
    # Micro F1 Score
    if average == 'micro':
        tp = np.sum(cm.diagonal())
        fp = np.sum(cm.sum(axis=0) - cm.diagonal())
        fn = np.sum(cm.sum(axis=1) - cm.diagonal())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Macro F1 Score
    elif average == 'macro':
        precision = np.mean([cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0 for i in range(len(cm))])
        recall = np.mean([cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0 for i in range(len(cm))])
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    
    return precision,recall,f1

## ***********  Deprecated  *********************

    
    # def confusMatrix(self):
    #     target = pd.Series(self.cat_target, name='Target')
    #     predict= pd.Series(self.cat_predict,name='Predicted')

    #     # creating confusion matrix
    #     df_confusion = pd.crosstab(target, predict)
    #     if len(self.labels) > 0:
    #         col_labels = [self.labels[i] for i in np.unique(predict)]
    #         df_confusion.columns = col_labels
    #         df_confusion.index = self.labels
    #     print('\n\t Confusion Matrix')
    #     print(df_confusion)
    
    # def getClassMetrics(self,bin_class=False):
    #     self.confusMatrix()
    #     if bin_class:
    #         recall = recall_score(self.cat_target,self.cat_predict)
    #         precision = precision_score(self.cat_target,self.cat_predict)
    #         f1 = f1_score(self.cat_target,self.cat_predict)
    #         print('Precision: {:.2f}%, recall: {:.2f}%, F1 score: {:.2f}%'.format(precision*100,recall*100,f1*100))

    #     else:
    #         print('\n\t Classification Report')
    #         print(classification_report(self.cat_target,self.cat_predict,target_names=self.labels))
    
    # def get_predict_metrics(self):
    #     invNorm = 9.89
    #     epsilon = 0.01
    #     zeros_index = np.where(self.y < epsilon)[0]
    #     y = np.delete(self.y,zeros_index)
    #     y_hat = np.delete(self.y_hat,zeros_index)

    #     #scaling back the sets from normalization (inverse normalization)
    #     y = np.array(y*invNorm)
    #     y_hat = np.array(y_hat*invNorm)

    #     #computing rmse and mape
    #     mape = np.mean(np.abs((y - y_hat) / y)) * 100
    #     rmse = np.sqrt(np.mean(np.power((y - y_hat),2)))
        
    #     return rmse, mape
        