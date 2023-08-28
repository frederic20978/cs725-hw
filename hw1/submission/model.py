import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d+1, self.num_classes))
        self.v = 0
    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        sigma = 1/(1 + np.exp(-x))
        return sigma

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        N = input_x.shape[0]
        input_x = np.hstack((input_x,np.ones((N,1))))
        output_z = np.dot(input_x,self.weights)
        predictedOutput = self.sigmoid(output_z)
        input_y = input_y[:,np.newaxis]
        loss = (input_y*np.log(predictedOutput))+((1-input_y)*np.log(1-predictedOutput))
        averageLoss = -np.sum(loss)/N
        return averageLoss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N = input_x.shape[0]
        input_x = np.hstack((input_x,np.ones((N,1))))
        output_z = np.dot(input_x,self.weights)
        predictedOutput = self.sigmoid(output_z)
        input_y = input_y[:,np.newaxis]
        error = predictedOutput - input_y
        gradient = np.dot(input_x.T,error)/N
        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        self.v = momentum*self.v - (learning_rate*(grad))
        self.weights = self.weights + self.v
        return self.weights

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        N = input_x.shape[0]
        input_x = np.hstack((input_x,np.ones((N,1))))
        output_z = np.dot(input_x,self.weights)
        predictions = self.sigmoid(output_z)
        predictions = predictions[:,0]
        predictions = (predictions>=0.5).astype(int)
        return predictions

class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3 # 3 classes
        self.d = 4 # 4 dimensional features
        self.weights = np.zeros((self.d+1, self.num_classes))
        self.v = 0
    
    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        sigma = 1/(1+np.exp(-x))
        return sigma

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        N = input_x.shape[0]
        input_x = np.hstack((input_x,np.ones((N,1))))
        z = np.dot(input_x,self.weights)
        predictions = self.sigmoid(z)
        trueLabel = np.zeros((N,self.num_classes))
        for value in range(self.num_classes):
            trueLabel[:,value] = (input_y==value).astype(int)
        loss = ((trueLabel*np.log(predictions))+ (1-trueLabel)*np.log(1-predictions))
        averageLoss = -1*np.sum(loss)/N
        return averageLoss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N = input_x.shape[0]
        input_y = input_y[:,np.newaxis]
        input_x = np.hstack((input_x,np.ones((N,1))))

        gradientMatrix = np.zeros_like(self.weights)

        for cls in range(self.num_classes):
            currentWeight = self.weights[:,cls]
            currentWeight = currentWeight[:,np.newaxis]
            z = np.dot(input_x,currentWeight)
            prediction = self.sigmoid(z)
            output_y = (input_y==cls).astype(int)
            error = prediction - output_y
            gradientMatrix[:,cls] = (np.dot(input_x.T,error)/N).flatten()

        return gradientMatrix

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        self.v = momentum*self.v - (learning_rate*grad)
        self.weights = self.v + self.weights
        return self.weights

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        N = input_x.shape[0]
        input_x = np.hstack((input_x,np.ones((N,1))))

        predictions = np.zeros((self.num_classes,N))
        z = np.dot(input_x,self.weights)
        predictions = self.sigmoid(z)
        classPredictions = np.argmax(predictions,axis = 1)
        return classPredictions
