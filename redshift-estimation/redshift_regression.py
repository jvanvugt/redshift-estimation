"""
@author: Joris van Vugt
"""

from sklearn.ensemble import RandomForestRegressor

class RedshiftRegressor(object):
    """
    Regression using random forests.
    """
    def __init__(self, n=8):
        self.regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.regressor.predict(X)
        return predictions
