## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling

# Read data matrix X and labels y from text file.
def read_data(file_name):
    
    #  YOUR CODE HERE
    X = []
    y = []
    
    data = np.loadtxt(file_name)
    
    
    X = data[:, 0]  # First column X
    y = data[:, 1]  # Second column y
            
    # Update X by adding a column of ones for the bias term
    X = np.array(X).reshape(-1, 1)
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones at the beginning of X
    return X, np.array(y)

# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):
    
    #  YOUR CODE HERE
    w = np.zeros(X.shape[1])  # Initialize w vector according to the number of columns in X
    costs = []  # List to hold cost values
    rmse_values = []  # List to hold RMSE values for each epoch

    for i in range(epochs):
        grad = compute_gradient(X, y, w)
        w -= lamda * grad  # Update weights
        cost = compute_cost(X, y, w)  # Calculate the current cost
        costs.append(cost)  # Append the cost to the list

        # Calculate RMSE for the current weights and add to the list
        rmse = compute_rmse(X, y, w)
        rmse_values.append(rmse)

    return w, costs, rmse_values

# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):
    
    #  YOUR CODE HERE
    total_squared_error = 0
    
    # Iterate through each sample in X
    for i in range(len(X)):
        # Manually calculate the function_value by iterating over elements
        function_value = 0
        for j in range(len(w)):
            function_value += X[i][j] * w[j]  # Element-wise product and sum
        
        # Calculate the error between function_value and actual value
        error = function_value - y[i]
        
        # Sum of squared error
        total_squared_error += error ** 2
    
    # Calculate Mean Squared Error (MSE)
    mse = total_squared_error / len(X)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    return rmse

# Compute objective (cost) function.
def compute_cost(X, y, w):
    
    #  YOUR CODE HERE
    total_cost = 0
    
    # Iterate through each sample in X
    for i in range(len(X)):
        # Manually calculate the function_value by iterating over elements
        function_value = 0
        for j in range(len(w)):
            function_value += X[i][j] * w[j]  # Element-wise product and sum
        
        # Calculate the error between function_value and actual value
        error = function_value - y[i]
        
        # Sum of squared error
        total_cost += error ** 2
    
    # Calculate average cost
    cost = (1 / (2 * len(X))) * total_cost
    return cost


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
    #  YOUR CODE HERE
    grad = np.zeros(w.shape)  # Initialize gradient as a zero vector
    
    # Iterate through each sample in X
    for i in range(len(X)):
        # Manually calculate the function_value by iterating over elements
        function_value = 0
        for j in range(len(w)):
            function_value += X[i][j] * w[j]  # Element-wise product and sum
        
        # Calculate the error between function_value and actual value
        error = function_value - y[i]
        
        # Update gradient
        for j in range(len(w)):
            grad[j] += error * X[i][j]  # Element-wise gradient update
    
    # Divide by the number of samples to get the average
    grad = grad / len(X)
    return grad


##======================= Main program =======================##

# Read the training and test data.
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

#  YOUR CODE HERE

# Standardize the data
mean, std = scaling.mean_std(Xtrain[:, 1:])  # Exclude bias column for standardization
Xtrain[:, 1:] = scaling.standardize(Xtrain[:, 1:], mean, std)
Xtest[:, 1:] = scaling.standardize(Xtest[:, 1:], mean, std)

# Train weights using gradient descent
lamda = 0.1
epochs = 500
w, costs, rmse_values = train(Xtrain, ttrain, lamda, epochs)

print("Values of w:")
print("w0 : ", w[0])
print("w[1]: ", w[1])


# Calculate weights using normal equations
# Using the formula: w = (X^T * X)^(-1) * X^T * y

w_normal = np.linalg.inv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(ttrain)

print("Weights from Normal Equations:")
print("w0 (Normal): ", w_normal[0])
print("w[1] (Normal): ", w_normal[1])

train_rmse = compute_rmse(Xtrain, ttrain, w)
test_rmse = compute_rmse(Xtest, ttest, w)

print("RMSE Values for both Train and Test Datas")
print("Train Rmse :",train_rmse)
print("Test RMSE :",test_rmse)
# Plot the cost function (J(w)) vs. the number of epochs
plt.plot(range(epochs), costs, color="blue")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost Function J(w)")
plt.title("Cost Function (J(w)) vs. Number of Iterations")
plt.show()

# Plot RMSE values vs. the number of epochs
plt.plot(range(epochs), rmse_values, color="purple")
plt.xlabel("Number of Iterations")
plt.ylabel("RMSE")
plt.title("RMSE vs. Number of Iterations")
plt.show()

# Plot training data with regression line (unchanged)
plt.figure(figsize=(18, 6))
# Training data scatter plot
plt.subplot(1, 2, 1)
plt.scatter(Xtrain[:, 1], ttrain, color="blue", label="Training Data")
plt.plot(Xtrain[:, 1], Xtrain.dot(w), color="red", label="Regression Line")
plt.xlabel("Standardized Size Values of Training Data")
plt.ylabel("House Price")
plt.title("Training Data with Regression Line")
plt.legend()

# Test data scatter plot
plt.subplot(1, 2, 2)
plt.scatter(Xtest[:, 1], ttest, color="green", marker="x", label="Test Data")
plt.plot(Xtest[:, 1], Xtest.dot(w), color="orange", label="Regression Line")
plt.xlabel("Standardized Size Values of Test Data")
plt.ylabel("House Price")
plt.title("Test Data with Regression Line")
plt.legend()

plt.tight_layout()
plt.show()

# Combined plot of training and test data with their regression lines
plt.figure(figsize=(10, 6))

# Scatter plot for training data
plt.scatter(Xtrain[:, 1], ttrain, color="blue", label="Training Data", alpha=0.6)

# Scatter plot for test data
plt.scatter(Xtest[:, 1], ttest, color="green", marker="x", label="Test Data", alpha=0.6)

# Regression line for training data
plt.plot(Xtrain[:, 1], Xtrain.dot(w), color="red", label="Regression Line (Train)", linewidth=2)

# Regression line for test data (using the same weights)
plt.plot(Xtest[:, 1], Xtest.dot(w), color="orange", label="Regression Line (Test)", linewidth=2)

plt.xlabel("Standardized Size Values of Both Training and Test Data")
plt.ylabel("House Price")
plt.title("Training and Test Data with Their Regression Lines")
plt.legend()
plt.grid(True)
plt.show()


# Create a bar graph to compare the parameters
labels = ['w0', 'w1']
values_gd = [w[0], w[1]]  # Parameters from Gradient Descent
values_ne = [w_normal[0], w_normal[1]]  # Parameters from Normal Equation

x = np.arange(len(labels))  # x positions
width = 0.35  # Width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values_gd, width, label='Gradient Descent')
bars2 = ax.bar(x + width/2, values_ne, width, label='Normal Equation')

# Graph properties
ax.set_ylabel('Parameter Values')
ax.set_title('Comparison of w Parameters')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

