import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros(num_features + 1)  # Initialize weights and bias to 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_gradient(self, input_x, input_y):
        N = input_x.shape[0]  # Number of samples
        ones_column = np.ones((N, 1))
        input_x = np.hstack((ones_column, input_x))  # Add a column of ones for bias
        
        z = np.dot(input_x, self.weights)  # Linear combination of inputs and weights
        predictions = self.sigmoid(z)
        
        error = predictions - input_y  # Difference between predicted and actual labels
        
        gradient = np.dot(input_x.T, error) / N  # Calculate gradient
        
        return gradient

# Example usage:
num_features = 2
model = LogisticRegression(num_features)

input_x = np.array([[2, 3], [1, 1], [5, 6]])
input_y = np.array([1, 0, 1])

gradient = model.calculate_gradient(input_x, input_y)
print("Gradient:", gradient)

