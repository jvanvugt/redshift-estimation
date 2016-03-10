"""
@author: Joris van Vugt
"""

from sklearn.neighbors.regression import KNeighborsRegressor

class RedshiftRegressor(object):
    """
    Regression using k-nearest neighbours.
    """
    def __init__(self):
        self.regressor = KNeighborsRegressor(n_neighbors=8)

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.regressor.predict(X)
        return predictions
