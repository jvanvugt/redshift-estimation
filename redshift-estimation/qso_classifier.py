"""
@author: Joris van Vugt
"""

from sklearn.ensemble import RandomForestClassifier

class QsoClassifier(object):
    """
    Classifier using random forests.
    """
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self


    def predict(self, X):
        predictions = self.classifier.predict(X)
        return predictions
