# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import counter
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features (not to a raw Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.
        
        trainingData and validationData are lists of feature Counters.    The corresponding
        label lists contain the correct label for each datum.
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """    
            
        # might be useful in your code later...
        # this is a list of all features in the training set.

        pre_fs = []
        for datum in trainingData:
            for key in datum.keys():
                pre_fs.append(key)

        
        feature_set = set(pre_fs)
        self.features = list(feature_set)

        #sys.exit(1)
        
        spgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

        Pr_distribution_of_features = counter.Counter()

        for label in trainingLabels:
            Pr_distribution_of_features[label] += 1

        Pr_distribution_of_features.normalize()
        self.Pr_distribution_of_features = Pr_distribution_of_features
        
        # Initialize stuff
        binary_feature_counts = {}
        totals = {}
        for f in self.features:
            binary_feature_counts[f] = {0: counter.Counter(), 1: counter.Counter()}
            totals[f] = counter.Counter()
                     
        # Calculate totals and binary feature counts

        # enumerate(thing) returns an iteratur that will return
        # (0, thing[0]), (1, thing[1]), (2, [thing[2]), ...
        for i, datum in enumerate(trainingData):
            y = trainingLabels[i]
            for f, value in datum.items():
                binary_feature_counts[f][value][y] += 1.0
                totals[f][y] += 1.0 
                
        bestConditionals = {}
        bestAccuracy = None
        # Evaluate each k, and use the one that yields the best accuracy
        for sp in spgrid or [0.0]:
            correct = 0
            conditionals = {}            
            for f in self.features:
                conditionals[f] = {0: counter.Counter(), 1: counter.Counter()}
                
            # Run Laplace smoothing
            for f in self.features:
                for value in [0, 1]:
                    for y in self.legalLabels:
                        conditionals[f][value][y] = (binary_feature_counts[f][value][y] + sp) / (totals[f][y] + sp*2)
                
            # Check the accuracy associated with this k
            self.conditionals = conditionals              
            guesses = self.classify(validationData)
            for i, guess in enumerate(guesses):
                correct += (validationLabels[i] == guess and 1.0 or 0.0)
            accuracy = correct / len(guesses)
            
            # Keep the best k so far
            if accuracy > bestAccuracy or bestAccuracy is None:
                bestAccuracy = accuracy
                bestConditionals = conditionals
                self.sp = sp
                
        self.conditionals = bestConditionals
                
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses
            
    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.        
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        logJoint = counter.Counter()
        evidence = datum.items()
        "*** YOUR CODE HERE ***"
        for y in self.legalLabels:
            if self.Pr_distribution_of_features[y] != 0:
                logJoint[y] = math.log(self.Pr_distribution_of_features[y])
            else:
                logJoint[y] = 0
            for f in self.conditionals:
                prob = self.conditionals[f][datum[f]][y]
                logJoint[y] += (prob and math.log(prob) or 0.0)

        return logJoint
 

        
            
