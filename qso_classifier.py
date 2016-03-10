from sklearn.neighbors import KNeighborsClassifier

class QsoClassifier(object):

    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=8)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self


    def predict(self, X):
        predictions = self.classifier.predict(X)
        return predictions
