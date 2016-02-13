import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

class QsoClassifier(object):
    
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=8)
    
    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self
    
    
    def predict(self, features, labels):
        predictions = self.classifier.predict(features)
        
        return predictions

    
