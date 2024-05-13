#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


'''

NAIVE BAYESIAN CLASSIFIER: Functions: train, predict

'''

# Naive Baesian Classification


class NaiveBayesClassifier:
    #Initialize probability sets
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        
    def train(self, X, y):
        num_samples = len(X)
        num_features = len(X[0])
        
        # Calculate class probabilities
        for label in set(y):
            self.class_probabilities[label] = sum(1 for item in y if item == label) / num_samples
        
        # Calculate feature probabilities
        for feature_index in range(num_features):
            feature_values = [sample[feature_index] for sample in X]
            for label in set(y):
                label_indices = [i for i in range(num_samples) if y[i] == label]
                feature_given_label = [feature_values[i] for i in label_indices]
                count_feature_given_label = sum(1 for value in feature_given_label if value == 1)
                self.feature_probabilities[(feature_index, label)] = count_feature_given_label / len(label_indices)
                
    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = float('-inf')
            predicted_class = None
            for label in self.class_probabilities:
                class_prob = self.class_probabilities[label]
                feature_probs = [self.feature_probabilities.get((i, label), 0) if sample[i] == 1 else 1 - self.feature_probabilities.get((i, label), 0) for i in range(len(sample))]
                total_prob = class_prob * reduce(lambda x, y: x * y, feature_probs)
                if total_prob > max_prob:
                    max_prob = total_prob
                    predicted_class = label
            predictions.append(predicted_class)
        return predictions

    
classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)

predictions = classifier.predict(X_test)

print(
    "Accuracy of Naive Bayesian Classifier:\n"
    f"{metrics.accuracy_score(y_test, predictions)}\n"
)


# In[ ]:




