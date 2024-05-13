#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[ ]:





# In[ ]:


'''

KNN Classifier: Functions: train, predict
'k' is variable

'''

# Nearest Neighbors Classification

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = []
            for train_sample, train_label in zip(self.X_train, self.y_train):
                distance = np.linalg.norm(sample - train_sample)
                distances.append((train_label, distance))
            distances.sort(key=lambda x: x[1])
            k_nearest = distances[:self.k]
            k_nearest_labels = [label for label, _ in k_nearest]
            predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(predicted_label)
        return predictions


N_values = [1, 5, 10, 20, 50, 100]

train_errors = []
test_errors = []

for N in N_values:
    knn = KNNClassifier(k=N)
    knn.train(X_train, y_train)
    
    predictions_test = knn.predict(X_test)
    predictions_train = knn.predict(X_train)

    train_error = 1 - metrics.accuracy_score(y_train, predictions_train)
    train_errors.append(train_error)

    test_error = 1 - metrics.accuracy_score(y_test, predictions_test)
    test_errors.append(test_error)


    
plt.figure(figsize=(10, 6))
plt.plot(N_values, train_errors, marker='o', label='Training Error Rate')
plt.plot(N_values, test_errors, marker='o', label='Test Error Rate')
plt.title('KNN Classifier Accuracy as # of Neighbors Increase')
plt.xlabel('# of Neighbors')
plt.ylabel('Error Rate')
plt.xticks(N_values)
plt.legend()
plt.grid(True)
plt.show()
   

