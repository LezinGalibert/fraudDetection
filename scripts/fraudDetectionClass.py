from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import tree
from joblib import dump, load
import math as m
import pandas as pd
import numpy as np

class fraudDetectionData():
    
    # Helper class to handle the labelled and unlabelled data from a unique dataset.
    
    # labelledData: labelled data within the dataset (L)
    # unlabelledData: unlabelled data within the dataset (U)
    # labels: label column of the dataset (Y)
    # isLabelled: helper pd.Series to remember whether the data were initially labelled or not
    # norm : choice of norm for normalizing the dataset.
    
    def __init__(self):
            self.labelledData = pd.DataFrame()
            self.unlabbeledData = pd.DataFrame()
    
            self.labels = pd.Series()
            self.isLabelled = pd.Series()

            self.norm = 'l1'
            
    def _fraudDetectedToInt_(self, labels):
        
        # Converts string data to integers.
        
        tmp = labels
        tmp = tmp.apply(lambda s : 1 if s == 'LEGIT'else (-1 if s == 'FRAUD' else 0))
        return tmp.copy()
        
    
    def importData(self, data, labels):
        
        # Splits the dataset in labelled and unlabelled data.
        # Assumes that the dataset has columns: {amount, total_amount_14days, email_handle_length,
        # email_handle_dst_char, total_nb_orders_player, player_seniority, total_nb_play_sessions,
        # geographic_distance_risk}, and that the labels have already been extracted from the 
        # initial dataset.
        
        tmp = data.copy()
        self.labels = self._fraudDetectedToInt_(labels)
        
        # order_created_datetime has been deemed too complicated to deal with at this stage.
        
        tmp.drop(['user_id', 'order_created_datetime'], axis='columns', inplace=True)
        
        cols = tmp.columns

        #Assumes uniform distribution of unlabelled data in the dataset,
        #could be refined with rearranging unlabelled data in training/test sets.
        
        self.labelledData   = tmp[self.labels != 0]
        self.unlabelledData = tmp[self.labels == 0]
        
        self.isLabelled = self.labels != 0
    
    def normalizeData(self):
        
        # Normalize the dataset using 'norm'.
        
        cols = self.labelledData.columns
       
        tmp = pd.concat([self.labelledData, self.unlabelledData])
        tmp_norm = pp.normalize(tmp.values, self.norm, axis=0)
        tmp = pd.DataFrame(data = tmp_norm, columns = cols, index = tmp.index)
        
        self.labelledData = tmp.loc[self.labelledData.index]
        self.unlabelledData = tmp.loc[self.unlabelledData.index]
        
        return tmp.copy()
                
    def buildPseudoClasses(self):
        
        # Step 3: finds the nearest neightboor using L1 norm between all unlabelled data and the labelled data.
        # This step can take an extremely long time and parallelization should be looked at for future improvements.
        
        
        l = len(self.labelledData)
        u = len(self.unlabelledData)
        
        for i in range(u):
            
            m = float("inf")
            print('Calculating distances for unlabelled data#' + str(i) + ' ...')
            
            for j in range(l):
                dist = np.absolute(self.labelledData.iloc[j] - self.unlabelledData.iloc[i]).sum()
                
                if m > dist:
                    
                    idxMin = self.labelledData.index[j]
                    m = dist
                    
            idx = self.unlabelledData.index[i]                 
            self.labels.loc[idx] = self.labels.loc[idxMin]
            
    def getLabels(self):
        return self.labels
    
    def getIsLabelled(self):
        return self.isLabelled
    
    def getColumns(self):
        return self.labelledData.columns