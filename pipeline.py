from __future__ import division
import numpy as np

from fetch_sdss_data import fetch_data
from qso_classifier import QsoClassifier

def get_data():
    data = fetch_data()
    N = len(data)
    X = np.zeros((N, 4))
    X[:, 0] = data['dered_u'] - data['dered_g']
    X[:, 1] = data['dered_g'] - data['dered_r']
    X[:, 2] = data['dered_r'] - data['dered_i']
    X[:, 3] = data['dered_i'] - data['dered_z']
    
    y = np.zeros((len(data), 2))
    y[:, 0] = data['class']
    y[:, 1] = data['z']
    X, y = remove_missing(X, y)
    return X, y
    
def remove_missing(features, labels):
    not_missing = ~np.isnan(features).any(axis=1)
    return features[not_missing, :], labels[not_missing, :]

def shuffle(features, labels):
    shuffled_indices = np.arange(len(features))
    np.random.shuffle(shuffled_indices)
    return features[shuffled_indices, :], labels[shuffled_indices, :]

def classification_info(predictions, labels):
    correct = [0] * 3
    counts = np.bincount(labels)
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct[label] += 1
    
    print 'accuracy:', sum(correct) / len(labels)
    
    print 'Galaxies: %i of %i' % (correct[0], counts[0])
    print 'Quasars: %i of %i' % (correct[1], counts[1])
    print 'Stars: %i of %i' % (correct[2], counts[2])
    print 'Total: %i of %i' % (sum(correct), len(labels))

def run():
    print 'Getting Data...'
    X, y = get_data()
    N = len(X)
    
    X, y = shuffle(X, y)
    X_train = X[(N / 2):, :]
    class_train = y[(N / 2):, 0].astype(np.int)
    X_test = X[:(N / 2), :]
    class_test = y[:(N / 2), 0].astype(np.int)
    
    clf = QsoClassifier()
    print 'Training classifier...'
    clf.fit(X_train, class_train)
    print 'Testing classifier...'
    predictions = clf.predict(X_test, class_train)
    
    classification_info(predictions, class_test)
    
    
    
        
    
if __name__ == '__main__':
    run()

