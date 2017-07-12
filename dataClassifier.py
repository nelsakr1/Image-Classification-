# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import naiveBayes
import perceptron
import kNN
import samples
import sys
import counter

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = counter.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = counter.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.
    
    Use the printImage(<list of pixels>) function to visualize features.
    
    An example of use has been given to you.
    
    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as counter.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features 
    
    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """
    
    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
            prediction = guesses[i]
            truth = testLabels[i]
            if (prediction != truth):
                    print "==================================="
                    print "Mistake on example %d" % i 
                    print "Predicted %d; truth is %d" % (prediction, truth)
                    print "Image: "
                    print rawTestData[i]
                    break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def printImage(self, pixels):
            """
            Prints a Datum object that contains all pixels in the 
            provided list of pixels.    This will serve as a helper function
            to the analysis function you write.
            
            Pixels should take the form 
            [(2,2), (2, 3), ...] 
            where each tuple represents a pixel.
            """
            image = samples.Datum(None,self.width,self.height)
            for pix in pixels:
                try:
                        # This is so that new features that you could define which 
                        # which are not of the form of (x,y) will not break
                        # this image printer...
                        x,y = pix
                        image.pixels[x][y] = 2
                except:
                        print "new features:", pix
                        continue
            print image    

def default(str):
    return str + ' [Default: %default]'

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser    
    parser = OptionParser(USAGE_STRING)
    
    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['naiveBayes', 'perceptron', 'kNN'], default='naiveBayes')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--neighbors', help=default("Numbers of neighbors in k-Nearest Neighbors"), type="int", default=3)
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}
    
    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        featureFunction = basicFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        featureFunction = basicFeatureExtractorFace            
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)
        
    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = range(2)
        
    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.neighbors <= 0:
        print "Neighbors for kNN should be a positive integer (you provided: %d)" % options.neighbors
        print USAGE_STRING
        sys.exit(2)
        
    if(options.classifier == "naiveBayes"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    elif(options.classifier == "perceptron"):
        classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
    elif(options.classifier == "kNN"):
        classifier = kNN.kNNClassifier(legalLabels,options.neighbors)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING
        
        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage
    
    return args, options

USAGE_STRING = """
    USAGE:            python dataClassifier.py <options>
    EXAMPLES:     (1) python dataClassifier.py
                                    - trains the default naiveBayes classifier on the digit dataset
                                    using the default 100 training examples and
                                    then test the classifier on test data
                            (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f
                                    - would run the naive Bayes classifier on 1000 training examples
                                    on the faces dataset, would test the classifier on the test data
                                 """

# Main harness code

def runClassifier(args, options):

    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
            
    # Load data    
    numTraining = options.training
    numTest = options.test

    if(options.data=="faces"):
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
    else: # default is digits
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
        
    
    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)
    
    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] ) 
    # Run classifier
    runClassifier(args, options)
