import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter
        self.automaticTuning = False
        self.class_probs = {}  
        self.feature_probs = {}
        
    def setSmoothing(self, k):
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = list(trainingData[0].keys())

        if self.automaticTuning:
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        best_accuracy = 0.0
        best_k = None
        original_k = self.k  # Store the original k value

        for k in kgrid:
            # Train the classifier with Laplace smoothing
            self.k = k
            self.trainNaiveBayes(trainingData, trainingLabels)

            # Calculate accuracy on validation data
            accuracy = self.validate(validationData, validationLabels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                
                

        

    def trainNaiveBayes(self, trainingData, trainingLabels):
        self.class_probs = {label: 0.0 for label in self.legalLabels}
        self.feature_probs = {label: {feature: 0.0 for feature in self.features} for label in self.legalLabels}

        total_count = len(trainingLabels)

        # Collect class counts
        class_counts = {label: 0 for label in self.legalLabels}
        for label_ in trainingLabels:
            class_counts[label_] += 1

        # Calculate class probabilities
        for label in self.legalLabels:
            self.class_probs[label] = (class_counts[label] + self.k) / (total_count + self.k * len(self.legalLabels))

        # Calculate feature probabilities
        for label in self.legalLabels:
            label_data = [datum for datum, label_ in zip(trainingData, trainingLabels) if label_ == label]
            label_word_counts = {feature: 0 for feature in self.features}
            for datum in label_data:
                for feature, count in datum.items():
                    label_word_counts[feature] += count
            for feature in self.features:
                self.feature_probs[label][feature] = (label_word_counts[feature] + self.k) / (
                        sum(label_word_counts.values()) + self.k * len(self.features))

    def validate(self, validationData, validationLabels):
        correct = 0
        total = len(validationLabels)

        for datum, true_label in zip(validationData, validationLabels):
            predicted_label = self.classify([datum])[0]
            if predicted_label == true_label:
                correct += 1

        accuracy = correct / total
        return accuracy

    def classify(self, testData):
        guesses = []

        for datum in testData:
            max_prob = float("-inf")
            best_label = None

            for label in self.legalLabels:
                prob = math.log(self.class_probs[label])

                for feature, count in datum.items():
                    if feature in self.feature_probs[label]:
                        prob += count * math.log(self.feature_probs[label][feature])

                if prob > max_prob:
                    max_prob = prob
                    best_label = label

            guesses.append(best_label)

        return guesses

    def calculateLogJointProbabilities(self, datum):
        logJoint = {label: 0.0 for label in self.legalLabels}

        for label in self.legalLabels:
            logJoint[label] = math.log(self.class_probs[label])

            for feature, count in datum.items():
                if feature in self.feature_probs[label]:
                    logJoint[label] += count * math.log(self.feature_probs[label][feature])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        featuresOdds = []

        for feature in self.features:
            odds_ratio = (self.feature_probs[label1][feature] + self.k) / (self.feature_probs[label2][feature] + self.k)
            featuresOdds.append((feature, odds_ratio))

        featuresOdds.sort(key=lambda x: -x[1])
        return [feature for feature, _ in featuresOdds[:100]]
