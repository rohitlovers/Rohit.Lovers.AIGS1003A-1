# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import samples
import sys
import util

TEST_SET_SIZE = 140
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

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
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
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  mistake_count = 0  # Track the number of mistakes found
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if prediction != truth:
          print("===================================")
          print("Mistake on example %d" % i)
          print("Predicted %d; truth is %d" % (prediction, truth))
          print("Image:")
          print(rawTestData[i])
          mistake_count += 1
          
      if mistake_count >= 4:
          break  # Stop after finding the first four mistakes
            
            
            
        
 

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

def default(str):
  return str + ' [Default: %default]'

def readCommand(argv):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0:
      raise Exception('Command line input not understood: ' + str(otherjunk)
                     )
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print("--------------------")
  print("data:\t\t" + options.data)
  print("classifier:\t\t" + options.classifier)
  print("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    featureFunction = basicFeatureExtractorDigit    
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = list(range(10))
  else:
    legalLabels = list(range(2))
    
  if options.training <= 0:
      print("Training set size should be a positive integer (you provided: %d)" % options.training)
      print(USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    if (options.autotune):
        print("using automatic tuning for naivebayes")
        classifier.automaticTuning = True
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training

  rawTrainingData = samples.loadDataFile("trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  trainingLabels = samples.loadLabelsFile("traininglabels", numTraining)
  rawValidationData = samples.loadDataFile("validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  validationLabels = samples.loadLabelsFile("validationlabels", TEST_SET_SIZE)
  rawTestData = samples.loadDataFile("testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
  testLabels = samples.loadLabelsFile("testlabels", TEST_SET_SIZE)
    
  
  # Extract features
  print("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))
  
  # Conduct training and testing
  print("Training...")
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  print("Validating...")
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
  print("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)