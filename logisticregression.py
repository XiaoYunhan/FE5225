import pandas as pd


class LogisticRegression:

    def __init__(self, penalty='l1', *, tol=1e-4, C=0.1):
        """
        :param penalty: 'l1' or 'l2' norm for regularization
        :param tol: tolerance for optimization
        :param C: See https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        """
        # your code here

    def fit(self, X, y):
        """
        This function is called to train the model given training examples (X and y)
        :param X: two-dimensional array of size N x d, where N is the number of training examples
        :param y: labels for the corresponding features in X, it is a one-dimensional array of size N
        :return:
        """
        # your code here
        return self

    def predict(self, X):
        """
        Given new examples X, this function return prediction
        :param X: features of new examples, It is a two-dimensional array of size m x d, where m is the number of
                new examples
        :return: one dimensional array of size m, prediction of new examples
        """
        # your code here

    def predict_prob(self, X):
        """
        Given new examples X, this function return prediction probability
        :param X: features of new examples, It is a two-dimensional array of size m x d, where m is the number of
                new examples
        :return: one dimensional array of size m, prediction probability of new examples
        """
        # your code here


def build_model(dataset):

    # 1. split the dataset into training, validation and test dataset
    # your code here

    # 2. use cross-validation techniques to choose model hyper-parameters penalty ('l1' or 'l2') and C
    # your code here
    # optimal_penalty = ...
    # optimal_C = ...

    # 3. train model with optimal hyper-parameters
    model = LogisticRegression(penalty=optimal_penalty, C=optimal_C)
    # your code here

    # 4. test model performance with test dataset
    # your code here

    return model


def test():
    data_file = 'mydatafile'
    dataset = pd.read_csv(data_file)
    model = build_model(dataset)

    # you can use model for further prediction
