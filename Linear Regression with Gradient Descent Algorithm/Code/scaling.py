import numpy as np

# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  
    
  # YOUR CODE HERE
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])  
  
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)
        
  return mean, std

# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
    # YOUR CODE HERE
   return (X - mean) / std