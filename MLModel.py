from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


def load_dataset():
    """
    Loading Iris Dataset
    """
    iris = load_iris()
    return iris


def get_data_labels(iris):
    """
    Getting features and targets from the dataset
    """
    X = iris.data
    Y = iris.target
    return X, Y


def fit_model(X, Y):
    """
    Fitting our Model on the dataset
    """
    clf = GaussianNB()
    clf.fit(X, Y)
    return clf


def main():
    iris = load_dataset()
    X, Y = get_data_labels(iris)
    clf = fit_model(X, Y)
    return iris, clf
