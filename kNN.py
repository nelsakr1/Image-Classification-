import counter
import classificationMethod
import math
import sys
import numpy

class kNNClassifier:

    def __init__(self, legalLabels, nNeighbors):
        self.legalLabels = legalLabels
        self.type = "kNN"
        self.nNeighbors = nNeighbors

    def manhattanDistance(x, y):
        return sum(abs(a-b) for a,b in zip(x,y))

    #def manhattanDistance (xy1, xy2):
    #    return abs(xy1[0] - xy2[0]) + abs(xy[1] - xy[1])

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        # Similarity: Calculate the distance between two data instances
        distances = []

        #for i in range(len(trainingData)):
            #dist = manhattanDistance(, trainingData[i])
            #distances.append(trainingData[i], dist)
            

        # Neighbors: Locate k most similar data instances
        neighbors = []

        print "(train method for kNN not defined)"
        sys.exit(1)

    def classify(self, data):

	# Neighbors: Locate k most similar data instances
        guesses = []

        k = self.nNeighbors
        #if (k == 1):
         

        print "(classify method for kNN not defined)"
        sys.exit(1)
