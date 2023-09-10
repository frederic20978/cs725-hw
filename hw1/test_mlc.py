import numpy as np

class MulticlassLogisticRegression:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = np.zeros((num_classes, num_features + 1))  # Initialize weights and bias for each class
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_gradient(self, input_x, input_y):
        N = input_x.shape[0]  # Number of samples
        ones_column = np.ones((N, 1))
        input_x = np.hstack((ones_column, input_x))  # Add a column of ones for bias
        
        gradients = np.zeros_like(self.weights)
        
        for c in range(self.num_classes):
            target = (input_y == c).astype(int)  # Target labels for the current class
            z = np.dot(input_x, self.weights[c])  # Linear combination of inputs and weights
            predictions = self.sigmoid(z)
            error = predictions - target  # Difference between predicted and target labels
            gradients[c] = np.dot(input_x.T, error) / N  # Calculate gradient for the current class
        
        return gradients

# Example usage:
num_features = 2
num_classes = 3
model = MulticlassLogisticRegression(num_features, num_classes)

input_x = np.array([[2, 3], [1, 1], [5, 6]])
input_y = np.array([0, 1, 2])  # Multiclass labels

gradients = model.calculate_gradient(input_x, input_y)
print("Gradients:", gradients)
