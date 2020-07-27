import os
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import tree
from joblib import dump, load
import math as m
import pandas as pd
import numpy as np
from fraudDetectionClass import fraudDetectionData
from assembleAdaBoostClass import assembleAdaBoost

# Main data Pipeline

data = pd.read_csv('mle_fraud_test.csv', sep=';', index_col=0)

# For time sake, we limit the dataset to 500 data points, with about 20 confirmed fraud cases
# and about 30 blocked cases for diversity.

data = data.iloc[5500:6000]

X = data[[c for c in data.columns if c != 'transaction_status']]
Y = data['transaction_status']

# Split train/test data with ration 80/20.

XTrain, XTest, YTrain, YTest = ttsplit(X, Y, test_size=0.2)

importTrainData = fraudDetectionData()
importTestData  = fraudDetectionData()

importTrainData.importData(XTrain, YTrain)
importTestData .importData(XTest, YTest)

XTrainNorm = importTrainData.normalizeData()
XTestNorm   = importTestData.normalizeData()

# Longest part of the pipeline process, should be optimize in the future.

importTrainData.buildPseudoClasses()

YTrainFull = importTrainData.getLabels()
YTestFull  = importTestData.getLabels()

isLabelledTrain = importTrainData.getIsLabelled()
isLabelledTest = importTestData.getIsLabelled()

l = sum(isLabelledTrain == 1)
u = sum(isLabelledTrain == 0)

cols = importTrainData.getColumns()
print(cols)

assembleModel = assembleAdaBoost([1 for _ in range(l + u)], 0.5, l, u, 15)
assembleModel.fit(XTrainNorm, YTrainFull, isLabelledTrain)
mat = assembleModel.confusionMatrix(XTrainNorm, YTrainFull)

print('############ CONFUSION MATRIX: TRAINING SET ################')
print('################ Predicted positives # Predicted negatives #')
print('Real positives #' + str(mat['true positives']).rjust(20) + ' # ' + str(mat['false negatives']).rjust(20) + '#')
print('Real negatives #' + str(mat['false positives']).rjust(20) + ' # ' + str(mat['true negatives']).rjust(20) + '#')
print('############################################################')

print('\n')

mat = assembleModel.confusionMatrix(XTestNorm, YTestFull)

print('############ CONFUSION MATRIX: TEST SET ####################')
print('################ Predicted positives # Predicted negatives #')
print('Real positives #' + str(mat['true positives']).rjust(20) + ' # ' + str(mat['false negatives']).rjust(20) + '#')
print('Real negatives #' + str(mat['false positives']).rjust(20) + ' # ' + str(mat['true negatives']).rjust(20) + '#')
print('############################################################')

print('\n')


def predict(dictValues, selectedColumns=cols, model=assembleModel):
    
    # Wrapper function to allow input to be json file.
    
    x = np.array([float(dictValues[col]) for col in cols])
    x = x.reshape(1,-1)
    pred = model.predict(x)[0]
    
    # In our current implementatin, s represent the 'score' to be legit, so p has to be mirrored to reflect the
    # probability to be a fraud.
    
    proba = 1 - assembleAdaBoost.scoreToProba(pred)
    block = assembleAdaBoost.blockTransaction(proba, -15, dictValues['amount'])
    
    # return both the probability to be a fraud and what decision (block or not) we should make.
    
    return [proba, block]
