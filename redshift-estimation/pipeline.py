"""
@author: Joris van Vugt
"""

from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

from fetch_sdss_data import fetch_data
from qso_classifier import QsoClassifier
from redshift_regression import RedshiftRegressor

def get_data():
    """
    Fetch data and extract features.
    
    X has four attributes: 
        dered_u - dered_g
        dered_g - dered_r
        dered_r - dered_i
        dered_i - dered_z
    
    y has two attributes:
        class: 0 (galaxy), 1 (quasar),or 2 (star)
        redshift
    """
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
    """
    Remove any rows from the data that have missing data.
    """
    not_missing = ~np.isnan(features).any(axis=1)
    return features[not_missing, :], labels[not_missing, :]

def shuffle(features, labels):
    """
    Shuffle the data.
    """
    shuffled_indices = np.arange(len(features))
    np.random.shuffle(shuffled_indices)
    return features[shuffled_indices, :], labels[shuffled_indices, :]

def add_correct(acc, x):
    """
    x is a tuple of the prediction and label of an instance.
    if the label is equals to the prediction, increment acc[label] by 1.
    """
    prediction, label = x
    if prediction == label:
        acc[label] += 1
    return acc

def classification_info(predictions, labels):
    """
    Print some info about the performance of the classifier.
    """
    print '\nClassifier:'
    counts = np.bincount(labels)

    correct = reduce(add_correct, zip(predictions, labels), [0, 0, 0])

    print 'accuracy:', sum(correct) / len(labels)

    print 'Galaxies: %i of %i' % (correct[0], counts[0])
    print 'Quasars: %i of %i' % (correct[1], counts[1])
    print 'Stars: %i of %i' % (correct[2], counts[2])
    print 'Total: %i of %i' % (sum(correct), len(labels))

def regressor_info(predictions, actual):
    """
    Print info about the performate of the regression.
    """
    print '\nRegression:'
    rms = np.sqrt(np.mean((actual - predictions) ** 2))
    print "RMS error = %.2g" % rms

    axis_lim = np.array([-0.1, 7.0])

    ax = plt.axes()
    plt.scatter(actual, predictions, c='k', lw=0, s=4)
    plt.plot(axis_lim, axis_lim, '--k')
    plt.plot(axis_lim, axis_lim + rms, ':k')
    plt.plot(axis_lim, axis_lim - rms, ':k')
    plt.xlim(axis_lim)
    plt.ylim(axis_lim)

    plt.text(0.99, 0.02, "RMS error = %.2g" % rms,
        ha='right', va='bottom', transform=ax.transAxes,
        bbox=dict(ec='w', fc='w'), fontsize=16)

    plt.title('Photo-z: Nearest Neigbor Regression')
    plt.xlabel(r'$\mathrm{z_{spec}}$', fontsize=14)
    plt.ylabel(r'$\mathrm{z_{phot}}$', fontsize=14)
    plt.show()

def find_quasar_indices(classes):
    """
    Find the indices of quasars.
    
    Returns a boolean array: 
    [class == quasar for class in classes]
    """
    idx = np.equal(classes, np.ones((1, len(classes))))
    idx.shape = classes.shape
    return idx

def classify(X_train, X_test, y_train, y_test):
    """
    Train and test a k-neareast neighbours classifier
    """
    clf = QsoClassifier()
    print 'Training classifier...'
    clf.fit(X_train, y_train)
    print 'Testing classifier...'
    predictions_clf = clf.predict(X_test)
    classification_info(predictions_clf, y_test)
    print

def regress(X_train, X_test, y_train, y_test):
    """
    Train and test a k-neareast neighbours regressor 
    """
    regressor = RedshiftRegressor()
    print 'Training regressor...'
    regressor.fit(X_train, y_train)
    print 'Testing regressor...'
    predictions_regressor = regressor.predict(X_test)
    regressor_info(predictions_regressor, y_test)

def run():
    """
    Train a classifier to find quasars and a regressor for estimating their
    redshifts
    """
    print 'Getting Data...'
    X, y = get_data()
    N = len(X)

    N_train = N / 2
    X, y = shuffle(X, y)
    X_train = X[N_train:, :]
    y_train = y[N_train:, :]
    X_test = X[:N_train, :]
    y_test = y[:N_train, :]

    class_train = y_train[:, 0].astype(np.int)
    class_test = y_test[:, 0].astype(np.int)

    # Predict the class of all objects
    classify(X_train, X_test, class_train, class_test)

    # Find the quasars in the data
    quasar_indices_train = find_quasar_indices(class_train)
    quasar_indices_test = find_quasar_indices(class_test)

    # Extract the quasars from the training and test set
    X_qso_train = X_train[quasar_indices_train]
    X_qso_test = X_test[quasar_indices_test]
    redshift_qso_train = y_train[quasar_indices_train, 1]
    redshift_qso_test = y_test[quasar_indices_test, 1]

    # Estimate redshift for quasars
    regress(X_qso_train, X_qso_test, redshift_qso_train, redshift_qso_test)



if __name__ == '__main__':
    run()
