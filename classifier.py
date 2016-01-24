import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

from fetch_sdss_data import fetch_data


def remove_missing(features, labels):
    not_missing = ~np.isnan(features).any(axis=1)
    return features[not_missing, :], labels[not_missing]    

def get_data():
    data = fetch_data()
    X = np.zeros((len(data), 4))
    
    X[:, 0] = data['dered_u'] - data['dered_g']
    X[:, 1] = data['dered_g'] - data['dered_r']
    X[:, 2] = data['dered_r'] - data['dered_i']
    X[:, 3] = data['dered_i'] - data['dered_z']
    y = data['class']
    X, y = remove_missing(X, y)
    
    X = preprocessing.scale(X)
    return X, y
    
def train(features, labels):
    classifier = KNeighborsClassifier()
    classifier.fit(features, labels)
    return classifier
    
def predict(features, labels, classifier):
    predictions = classifier.predict(features)
    correct = [0] * 3
    counts = np.bincount(labels)
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct[label] += 1
    
    print classifier.score(features, labels)
    
    print 'Galaxies: %i of %i' % (correct[0], counts[0])
    print 'Quasars: %i of %i' % (correct[1], counts[1])
    print 'Stars: %i of %i' % (correct[2], counts[2])
    print 'Total: %i of %i' % (sum(correct), len(labels))

def learn():
    X, y = get_data()    
    N = len(X)
    print 'N =', N
    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]
    
    X_train = X[(N / 2):, :]
    y_train = y[(N / 2):]
    X_test = X[:(N / 2), :]
    y_test = y[:(N / 2)]
    
    classifier = train(X_train, y_train)
    predict(X_test, y_test, classifier)
    
    
if __name__ == '__main__':
    learn()
