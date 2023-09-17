import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        self.number_of_classes = len(self.unique_classes)
        
        self.class_counts = np.zeros(self.number_of_classes)
        self.class_priors = {}
        for i, output_cls in enumerate(self.unique_classes):
            class_count = np.sum(y == output_cls)
            self.class_counts[i] = class_count
            self.class_priors[output_cls] = class_count/ len(y)

        self.gaussian_params = {}
        self.bernoulli_params = {}
        self.laplace_params = {}
        self.exponential_params = {}
        self.multinomial_params = {}

        for output_cls in self.unique_classes:
            X_cls = X[y == output_cls]
            
            # Gaussian distribution
            mean_of_x1 = np.mean(X_cls[:, 0])  # MLE for X1
            var_of_x1 = np.var(X_cls[:, 0],ddof=1)    # MLE for X1

            mean_of_x2 = np.mean(X_cls[:, 1])  # MLE for X2
            var_of_x2 = np.var(X_cls[:, 1],ddof=1)    # MLE for X2
            self.gaussian_params[output_cls] = (mean_of_x1, mean_of_x2, var_of_x1, var_of_x2)

            # Bernoulli distribution
            # calculatin mean is same as calculating probability of success as mean=p for bernoulli
            p_of_x3 = np.mean(X_cls[:, 2])     # MLE for X3
            p_of_x4 = np.mean(X_cls[:, 3])     # MLE for X4
            self.bernoulli_params[output_cls] = (p_of_x3, p_of_x4)

            # Laplace distribution
            median_of_x5 = np.median(X_cls[:, 4])  # MLE for X5
            b_of_x5 = np.mean(np.abs(X_cls[:, 4] - median_of_x5))  # MLE for X5
            median_of_x6 = np.median(X_cls[:, 5])  # MLE for X6
            b_of_x6 = np.mean(np.abs(X_cls[:, 5] - median_of_x6))  # MLE for X6
            self.laplace_params[output_cls] = (median_of_x5,median_of_x6, b_of_x5, b_of_x6)

            # Exponential distribution
            lambda_of_x7 = 1 / np.mean(X_cls[:, 6])  # MLE for X7
            lambda_of_x8 = 1 / np.mean(X_cls[:, 7])  # MLE for X8
            self.exponential_params[output_cls] = (lambda_of_x7, lambda_of_x8)

            # Multinomial distribution
            unique_categories, category_counts = np.unique(X_cls[:, 8], return_counts=True)  # For X9
            p_of_classes_x9 = category_counts / len(X_cls[:, 8])
            unique_categories, category_counts = np.unique(X_cls[:, 9], return_counts=True)  # For X10
            p_of_classes_x10 = category_counts / len(X_cls[:, 9])
            self.multinomial_params[output_cls] = (p_of_classes_x9, p_of_classes_x10)


    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []

            for output_cls in self.unique_classes:
                prior = self.class_priors[output_cls]
                likelihood = self.calculate_likelihood(x, output_cls)
                posterior = prior * likelihood
                posteriors.append(posterior)

            predicted_class = np.argmax(posteriors)
            predictions.append(predicted_class)

        return np.array(predictions)

    def calculate_likelihood(self, x, output_cls):
        # Gaussian distribution
        mean_of_x1, mean_of_x2, var_of_x1, var_of_x2 = self.gaussian_params[output_cls]
        gaussian_likelihood_x1 = (1 / np.sqrt(2 * np.pi * var_of_x1)) * np.exp((-(x[0] - mean_of_x1) ** 2) / (2 * var_of_x1))
        gaussian_likelihood_x2 = (1 / np.sqrt(2 * np.pi * var_of_x2)) * np.exp((-(x[1] - mean_of_x2) ** 2) / (2 * var_of_x2))

        # Bernoulli distribution
        p_of_x3, p_of_x4 = self.bernoulli_params[output_cls]
        bernoulli_likelihood_x3 = p_of_x3 ** x[2] * (1 - p_of_x3) ** (1 - x[2])
        bernoulli_likelihood_x4 = p_of_x4 ** x[3] * (1 - p_of_x4) ** (1 - x[3])

        # Laplace distribution
        median_of_x5, median_of_x6, b_of_x5, b_of_x6 = self.laplace_params[output_cls]
        laplace_likelihood_x5 = (1 / (2 * b_of_x5)) * np.exp(-np.abs(x[4] - median_of_x5) / b_of_x5)
        laplace_likelihood_x6 = (1 / (2 * b_of_x6)) * np.exp(-np.abs(x[5] - median_of_x6) / b_of_x6)

        # Exponential distribution
        lambda_of_x7, lambda_of_x8 = self.exponential_params[output_cls]
        exponential_likelihood_x7 = lambda_of_x7 * np.exp(-lambda_of_x7 * x[6])
        exponential_likelihood_x8 = lambda_of_x8 * np.exp(-lambda_of_x8 * x[7])

        # Multinomial distribution
        p_x9, p_x10 = self.multinomial_params[output_cls]
        multinomial_likelihood_x9 = p_x9[int(x[8])]
        multinomial_likelihood_x10 = p_x10[int(x[9])]

        return (
            gaussian_likelihood_x1 * gaussian_likelihood_x2 *
            bernoulli_likelihood_x3 * bernoulli_likelihood_x4 *
            laplace_likelihood_x5 * laplace_likelihood_x6 *
            exponential_likelihood_x7 * exponential_likelihood_x8 *
            multinomial_likelihood_x9 * multinomial_likelihood_x10
        )

    def getParams(self):
        priors = {}
        gaussian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        for output_cls in self.unique_classes:
            priors[str(output_cls)] = self.class_priors[output_cls]
            gaussian[str(output_cls)] = list(self.gaussian_params[output_cls])
            bernoulli[str(output_cls)] = list(self.bernoulli_params[output_cls])
            laplace[str(output_cls)] = list(self.laplace_params[output_cls])
            exponential[str(output_cls)] = list(self.exponential_params[output_cls])
            multinomial[str(output_cls)] = list(self.multinomial_params[output_cls])

        return (priors, gaussian, bernoulli, laplace, exponential, multinomial)
    
    
def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels,name):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Visualization of Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(name)
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_positives = np.sum((predictions == label) & (true_labels != label))
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_negatives = np.sum((predictions != label) & (true_labels == label))
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        
        prec = precision(predictions, true_labels, label)
        rec = recall(predictions, true_labels, label)
        return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
        """End of your code."""
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # params = model.getParams()
    # print(params)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # print(validation_datapoints.shape)
    visualise(validation_datapoints, validation_predictions, "validation_predictions.png")