#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''

NEURAL NETWORK WITH STOCHASTIC BACK-PROPAGATION
Input layer size and hidden layer size are variable/can be adjusted
Learning rate is variable/can be adjusted

In this example, I am testing the output from initializing the weights of the network randomly

Sigmoid function: f(x) = 1.716 tanh(2/3 x)

'''


import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1.716 * np.tanh(2/3 * x)

# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return 1.716 * (2/3) * (1 - np.tanh(2/3 * x)**2)

# Initialize weights randomly or with constants
def initialize_weights_random(input_size, hidden_size, output_size):
    w_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
    w_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
    return w_input_hidden, w_hidden_output

def initialize_weights_constant(input_size, hidden_size, output_size):
    w_input_hidden = np.full((input_size, hidden_size), 0.5)
    w_hidden_output = np.full((hidden_size, output_size), -0.5)
    return w_input_hidden, w_hidden_output

# Stochastic backpropagation
def backpropagation(X, y, w_input_hidden, w_hidden_output, learning_rate, epochs):
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # Forward pass
            hidden_input = np.dot(X[i], w_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output = sigmoid(np.dot(hidden_output, w_hidden_output))
            
            # Backward pass
            output_error = y[i] - output
            total_error += np.abs(output_error)
            output_delta = output_error * sigmoid_derivative(output)
            hidden_error = np.dot(output_delta, w_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_input)
            
            # Weight updates
            w_hidden_output += learning_rate * np.outer(hidden_output, output_delta)
            w_input_hidden += learning_rate * np.outer(X[i], hidden_delta)
            
        errors.append(total_error / len(X))
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error / len(X)}")
    return errors

# Define training data (this was provided in my assignment)
X = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87], [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38], [-0.76, 0.84, -1.96], 
              [0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16], [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14], [0.46, 1.49, 0.68]])
y = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]])

# Define network architecture
input_size = 3
hidden_size = 1
output_size = 1

# Parameters
learning_rate = 0.1
epochs = 1000

# Initialize weights randomly
w_input_hidden, w_hidden_output = initialize_weights_random(input_size, hidden_size, output_size)
errors_random = backpropagation(X, y, w_input_hidden, w_hidden_output, learning_rate, epochs)

# Initialize weights with constants
w_input_hidden, w_hidden_output = initialize_weights_constant(input_size, hidden_size, output_size)
errors_constant = backpropagation(X, y, w_input_hidden, w_hidden_output, learning_rate, epochs)

# Plot learning curves
plt.plot(range(epochs), errors_random, label='Random Initialization')
plt.plot(range(epochs), errors_constant, label='Constant Initialization')
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.title('Learning Curves')
plt.legend()
plt.show()

'''
Although there is extremely little difference between the constant initialization and the random initialization, we can 
attribute this difference to the random weight initialization. The random weigh initialization allows the network to 
explore different parts of the weight space during training.
'''


# In[ ]:





# In[ ]:




