from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import tree
from joblib import dump, load
import math as m
import pandas as pd
import numpy as np

# Class used to build the ASSEMBLE.AdaBoost model from http://homepages.rpi.edu/~bennek/kdd-KristinBennett1.pdf

class assembleAdaBoost():
    # alpha: weigth vector used for the miscalculation cost
    # beta: parameter for the initialization of the first miscalculation cost
    # lab: number of labelled data (l)
    # ulab: number of unlabelled data (u)
    # modelNumber: number of models to take into account in our training (T)
    # classifiers: list of weak models to use (ft). Here we use the same decision tree model with
    #              depth limited to 3 (log2(n_features)).
    # labelEncoders: as we update the unlabelled data (yi = Ft(xi) for i in U), yi becomes a float and
    #                potentially different from 1 / -1. This list allows to encode the new labels for a
    #                multiclass classification by the weak learners.
    # misclassificationCost: misclassification cost at the current iteration (Dt).
    # weights: weigths used to for the ensemble model (wi).
    # currPrediction: place holder for the current prediction data (Ft(x)).
    # convergence: convergence at each iteration step (epsilon).
    
    def __init__(self, a, b, l, u, T):
        self.alpha = a
        self.beta = b
        self.lab = l
        self.ulab = u
        self.modelNumber = T
        
        self.classifiers = [tree.DecisionTreeClassifier(random_state=0, max_depth=3) for _ in range(T)]
        self.labelEncoders = [pp.LabelEncoder() for _ in range(T)]
        self.misclassificationCost = [0 for _ in range(self.lab + self.ulab)]
        self.weigths = [0 for _ in range(T)]
    
        self.currPrediction = [0 for _ in range(self.lab + self.ulab)]
        self.convergence = [0 for _ in range(T)]
        
    def _initMisclassificationCost_(self, isLabelled):
        
        # Step 2: initialize the miscalculation cost. isLabelled provides help knowing which data was initially
        #         labelled or not.
        
        for i in range(self.ulab + self.lab - 1):
            self.misclassificationCost[i] = [self.beta/self.lab, (1 - self.beta)/self.ulab][isLabelled.iloc[i] == 0]
            
        # Due to rounding errors, the sum of the array tends to be slightly below 1, which triggers errors when calling the
        # sampling function. The next step helps getting around this problem while maintaining relative consistency with the
        # actual distribution. The last item of the list is replaced by 1 minus the sum of all previous elements.
            
        s = sum(self.misclassificationCost[:-1])
        self.misclassificationCost[-1] = 1 - s
        
    def _sampleFromDistribution_(self):
        
        # Step 13 (and part of step 4): samples l + u data points from the list [1..l+u], each point weigthed by Dt.
        
        return np.random.choice(np.arange(self.lab + self.ulab),
                                self.lab,
                                p = self.misclassificationCost)
    
    def _fitWeakLearner_(self, X, labels, t):
        
        # Step 4 and 14: fit the t th weak learning. Labels are saved for retrieving the correct predictions later.
        
        encodedLabels = self.labelEncoders[t].fit_transform(labels)
        self.classifiers[t].fit(X, encodedLabels)
        
    
    def _initModel_(self, X, labels, isLabelled):
        
        # Step 2 and 4: initialize the model.
        
        self._initMisclassificationCost_(isLabelled)
        
        S = self._sampleFromDistribution_()
        
        self._fitWeakLearner_(X.iloc[S], labels.iloc[S], 0)
        
        
    def _predictFromWeakLearners_(self, X, i):
        
        # Step 10: use the i first weak learners and weigths to build the predictions. This is mainly used for
        #.         building the misclassification cost.
        
        # Clears out current prediction table
        
        self.currPrediction = [0 for _ in range(self.lab + self.ulab)]

        for m in range(i):
            
            # Here we used the m th stored labels to decode prediction of the m th weak learner. Otherwise,
            # We would only predict with the encoded labels.
            encodedPrediction = self.classifiers[m].predict(X)
            weakPrediction = self.labelEncoders[m].inverse_transform(encodedPrediction)
            
            self.currPrediction = [ x + self.weigths[m]*y for x, y in zip(self.currPrediction, weakPrediction)]
        
            
    def _updateMisclassificationCost_(self, labels):
        
        # Step 12: update Dt.
        
        for i in range(len(self.misclassificationCost)):
            self.misclassificationCost[i] = self.alpha[i] * m.exp(-labels.iloc[i] * self.currPrediction[i])
            
        s = sum(self.misclassificationCost)
        
        for i in range(len(self.misclassificationCost)):
            self.misclassificationCost[i] /= s
            
        # Same trick used as in _initMisclassificationCost_.
            
        s = sum(self.misclassificationCost[:-1])
        self.misclassificationCost[-1] = 1 - s
            

    def _updateLabels_(self, labels, isLabelled):
        
        # Step 11: update labels of initially unlabelled data based on the current prediction data.
        
        for i in range(len(labels)):
            if not isLabelled.iloc[i]:
                labels.iloc[i] = self.sgn(self.currPrediction[i])
            
        return labels
    
    def _calculateConvergence_(self, X, labels, i):
        
        # Steps 6 and 7: build a match/don't match list based on the prediction of the i th weak learner and
        #                calculate the convergence epsilon based on the current Dt.
        
        model = self.classifiers[i]
        
        weakEncodedPrediction = model.predict(X)
        weakPrediction = self.labelEncoders[i].inverse_transform(weakEncodedPrediction)
        
        correctPrediction = [weakPrediction[i] == labels.iloc[i] for i in range(len(labels))]
        
        self.convergence = sum([self.misclassificationCost[k] for k in range(self.lab + self.ulab)
                                                          if weakPrediction[k] != labels.iloc[k]])
        
        
    def _updateWeight_(self, i):
        
        # Step 9: update i th weigths.
        
        self.weigths[i] = 0.5*m.log(1/self.convergence - 1)
        
        
    def writeModel(self):
        
        # Helper function in case we want to save our model for later...
        
        for m in range(self.modelNumber):
            dump(self.classifiers[m], 'trainedClassifier' + str(m) + '.model')
            dump(self.labelEncoders[m], 'encodedLabels' + str(m) + '.labels')
            
        np.save('weigths', self.weigths)
            
    def loadModel(self):
        
        # Helper function in case we want to save our model for later...
        
        for m in range(self.modelNumber):
            self.classifiers[m] = load('trainedClassifier' + str(m) + '.model')
            self.labelEncoders[m] = load('encodedLabels' + str(m) + '.labels')
            
        self.weigths = np.load('weigths.npy')
            
        
    def fit(self, X, labels, isLabelled):
        
        # Main fit function
        
        localLabels = pd.Series([int(d) for d in labels.values], index=labels.index, dtype=np.dtype("int32"))
           
        print("Fitting is starting!")
        print("Model initialization:")
        
        self._initModel_(X, localLabels, isLabelled)
        print("   Distribution initialized")
        print("   First weak learner trained")
        
        
        print("Looping through weak learners...")
        
        for t in range(1, self.modelNumber):
            self._calculateConvergence_(X, localLabels, t - 1)
            
            print("   Convergence for loop #" + str(t) + " is " + str(self.convergence))
            
            if self.convergence > 0.5:
                print("      Convergence criteria is " + str(self.convergence) + " greater than 0.5." )
                print("      Exiting training loop!")
                
                self.modelNumber = t
                
                break
                
            if self.convergence == 0.0:
                
                self.modelNumber = self.modelNumber - 1
                
                continue
                
            print("      Convergence criteria is" + str(self.convergence) + " smaller than 0.5." )
            print("      Continuing...")
            
            print("         Calculating weight...")
            self._updateWeight_(t - 1)
            
            print("         Updating predictions...")
            self._predictFromWeakLearners_(X, t)
            
            
            print("         Updating labels...")
            self._updateLabels_(localLabels, isLabelled)
            
            
            print("         Updating cost distribution...")
            self._updateMisclassificationCost_(localLabels)
            
            indices = self._sampleFromDistribution_()
            print("         Training new weak learner...")
            self._fitWeakLearner_(X.iloc[indices], localLabels.iloc[indices], t)
            
            print("         Loop ended, continuing...")
            
            
    def predict(self, X):
        
        # Main predict function.
        
        pred = [0 for _ in range(len(X))]

        for t in range(self.modelNumber):
            
            # Loop through all models and sums the predictions based on t th weak learner, encoded labels and weigth.

            encodedPrediction = self.classifiers[t].predict(X)
            weakPrediction = self.labelEncoders[t].inverse_transform(encodedPrediction)

            pred = [ x + self.weigths[t]*y for x, y in zip(pred, weakPrediction)]

        return pred
    
    def confusionMatrix(self, X, labels):
        
        pred = self.predict(X)
        matrix = {'true positives': 0, 'false positives': 0, 'true negatives': 0, 'false negatives': 0}
        
        for i in range(len(pred)):
            matrix['true positives' ] += int(self.sgn(pred[i]) ==  1 and labels.iloc[i] ==  1)
            matrix['false positives'] += int(self.sgn(pred[i]) ==  1 and labels.iloc[i] == -1)
            matrix['true negatives' ] += int(self.sgn(pred[i]) == -1 and labels.iloc[i] == -1)
            matrix['false negatives'] += int(self.sgn(pred[i]) == -1 and labels.iloc[i] ==  1)
                
        return matrix
    
    @staticmethod
    def sgn(s):
        if s >= 0:
            return 1
        else:
            return -1
    
    @staticmethod
    def scoreToProba(s):
        
        # Sigmoid function to convert s in R to p in [0, 1].
        
        return 1/(1 + m.exp(s))
    
    @staticmethod
    def blockTransaction(p, F, M):
        
        # Calculate the expectation of the event "Not blocking the order.
        # If the value is negative, then we loose money by not blocking,
        # and we therefore need to block the transaction. Otherwise, we should
        # not block the transaction.
        
        return - p * F +(1 - p) * M < 0
                