"""
@author: Joris van Vugt
"""

from sklearn.neighbors import KNeighborsClassifier

class QsoClassifier(object):
    """
    Classifier using k-nearest neighbours.
    """
    def __init__(self, n=8):
        self.classifier = KNeighborsClassifier(n_neighbors=n)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self


    def predict(self, X):
        predictions = self.classifier.predict(X)
        return predictions
