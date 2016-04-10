"""
@author: Joris van Vugt
"""

from __future__ import division
import sys
from math import sqrt, floor, ceil

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, confusion_matrix, \
    accuracy_score
from sklearn.cross_validation import train_test_split

from fetch_sdss_data import get_data_from_csv
from qso_classifier import QsoClassifier
from redshift_regression import RedshiftRegressor

def get_data():
    """
    Fetch data and extract features.

    X has 30 attributes:
        {dered, psfMag, modelMag, petroMag, fiberMag, extinction}_{u,g,r,i,z}

    y has two attributes:
        class: 0 (galaxy), 1 (quasar),or 2 (star)
        redshift
    """
    data = get_data_from_csv('SDSS_DR12_extra.csv')
    N = len(data)
    X = np.zeros((N, 30))
    X[:, 0] = data['dered_u']
    X[:, 1] = data['dered_g']
    X[:, 2] = data['dered_r']
    X[:, 3] = data['dered_i']
    X[:, 4] = data['dered_z']

    X[:, 5] = data['psfMag_u']
    X[:, 6] = data['psfMag_g']
    X[:, 7] = data['psfMag_r']
    X[:, 8] = data['psfMag_i']
    X[:, 9] = data['psfMag_z']

    X[:, 10] = data['modelMag_u']
    X[:, 11] = data['modelMag_g']
    X[:, 12] = data['modelMag_r']
    X[:, 13] = data['modelMag_i']
    X[:, 14] = data['modelMag_z']

    X[:, 15] = data['petroMag_u']
    X[:, 16] = data['petroMag_g']
    X[:, 17] = data['petroMag_r']
    X[:, 18] = data['petroMag_i']
    X[:, 19] = data['petroMag_z']

    X[:, 20] = data['fiberMag_u']
    X[:, 21] = data['fiberMag_g']
    X[:, 22] = data['fiberMag_r']
    X[:, 23] = data['fiberMag_i']
    X[:, 24] = data['fiberMag_z']

    X[:, 25] = data['extinction_u']
    X[:, 26] = data['extinction_g']
    X[:, 27] = data['extinction_r']
    X[:, 28] = data['extinction_i']
    X[:, 29] = data['extinction_z']

    y = np.zeros((len(data), 2))
    y[:, 0] = data['class']
    y[:, 1] = data['z']
    X, y = remove_missing(X, y)
    return X, y

def remove_missing(features, labels):
    """
    Remove all rows from the data that have missing data.
    """
    not_missing = ~np.isnan(features).any(axis=1)
    return features[not_missing, :], labels[not_missing, :]

def classification_info(predictions, labels, save_plots):
    """
    Print some info about the performance of the classifier.
    """
    print '\nClassifier:'

    plt.figure()
    cm = confusion_matrix(labels, predictions)
    normalized_cm = np.log10(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    names = ['Star', 'Quasar', 'Galaxy']
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Classification: normalized confusion matrix (logscale)')
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_plots:
        plt.show()
    else:
        plt.savefig('../plots/classification.png')

    print 'accuracy:', accuracy_score(labels, predictions)

def regressor_info(predictions, actual, save_plots):
    """
    Print info about the performance of the regression model.
    """
    print '\nRegression:'
    rms = sqrt(mean_squared_error(actual, predictions))
    print "RMS error = %.2g" % rms

    minimum = floor(min(min(predictions), min(actual))) - 0.1
    maximum = ceil(max(max(predictions), max(actual)))

    axis_lim = np.array([minimum, maximum])

    plt.figure()
    ax = plt.axes()
    plt.hexbin(actual, predictions, mincnt=1, gridsize=300)
    plt.plot(axis_lim, axis_lim, '--k')
    plt.plot(axis_lim, axis_lim + rms, ':k')
    plt.plot(axis_lim, axis_lim - rms, ':k')
    plt.xlim(axis_lim)
    plt.ylim(axis_lim)
    plt.colorbar()

    plt.text(0.99, 0.02, "RMS error = %.2g" % rms,
       ha='right', va='bottom', transform=ax.transAxes,
       bbox=dict(ec='w', fc='w'), fontsize=16)

    plt.title('Redshift estimation using random forests')
    plt.xlabel(r'$\mathrm{z_{spec}}$', fontsize=14)
    plt.ylabel(r'$\mathrm{z_{phot}}$', fontsize=14)
    if save_plots:
        plt.show()
    else:
        plt.savefig('../plots/regressor.png')

def find_redshift_indices(redshifts, value, unequality=np.greater_equal):
    """
    Return a list with True and False indicating where the redshift satisfies
    the unequality.

    unequality is a numpy function like np.greater, np.less, etc.
    """
    idx = unequality(redshifts, np.full((1, len(redshifts)), value, dtype=np.int))
    idx.shape = redshifts.shape
    return idx

def find_quasar_indices(classes):
    """
    Find the indices of quasars.

    Return a boolean array:
    [class == quasar for class in classes]
    """
    idx = np.equal(classes, np.ones((1, len(classes))))
    idx.shape = classes.shape
    return idx

def classify(X_train, X_test, y_train, y_test, save_plots):
    """
    Train and test a random forest classifier

    Return the predictions for the test set
    """
    clf = QsoClassifier()
    print 'Training classifier...'
    clf.fit(X_train, y_train)
    print 'Testing classifier...'
    predictions_clf = clf.predict(X_test)
    classification_info(predictions_clf, y_test, save_plots)
    print
    return predictions_clf

def regress(X_train, X_test, y_train, y_test, save_plots):
    """
    Train and test a random forest regressor
    Return the predictions for the test set
    """
    regressor = RedshiftRegressor()
    print 'Training regressor...'
    regressor.fit(X_train, y_train)
    print 'Testing regressor...'
    predictions_regressor = regressor.predict(X_test)
    regressor_info(predictions_regressor, y_test, save_plots)
    return predictions_regressor

def bin_by_redshift(regions, redshifts):
    """
    regions should be a list in ascending order.
    Return a list of len(redshifts) with a bin number for each instance
    The first bin will contain all instances where z < regions[0]
    The last bin will contain all instances where z > regions[-1]
    """
    regs = regions + [float('inf')]
    idx = np.zeros((len(redshifts)))
    for i, z in enumerate(redshifts):
        idx[i] = next(regs.index(r) for r in regs if r > z)
    return idx

def regress_per_region(regions, X_train, X_test, y_train, y_test, save_plots):
    bin_idx_train = bin_by_redshift(regions, y_train)
    bin_idx_test = bin_by_redshift(regions, y_test)

    for i in xrange(len(regions) + 1):
        X_train_region = X_train[bin_idx_train == i]
        X_test_region = X_test[bin_idx_test == i]
        y_train_region = y_train[bin_idx_train == i]
        y_test_region = y_test[bin_idx_test == i]
        regress(X_train_region, X_test_region, y_train_region, y_test_region,
                                                                    save_plots)



def run(save_plots=True):
    """
    Train a classifier to find quasars and a regressor for estimating their
    redshifts
    """
    print 'Getting Data...'
    X, y = get_data()
    N = len(X)
    print "N:", N

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                             random_state=42)
    class_train = y_train[:, 0].astype(np.int)
    class_test = y_test[:, 0].astype(np.int)

    # Predict the class of all objects
    # classify(X_train, X_test, class_train, class_test, save_plots)

    # Find the quasars in the data
    quasar_indices_train = find_quasar_indices(y_train[:, 0])
    quasar_indices_test = find_quasar_indices(y_test[:, 0])

    # Extract the quasars from the training and test set
    X_qso_train = X_train[quasar_indices_train]
    X_qso_test = X_test[quasar_indices_test]
    redshift_qso_train = y_train[quasar_indices_train, 1]
    redshift_qso_test = y_test[quasar_indices_test, 1]

    print "n_quasars:", len(X_qso_train) + len(X_qso_test)

    # Estimate redshift for quasars
    regress_per_region([2, 4], X_qso_train, X_qso_test, redshift_qso_train,
                                            redshift_qso_test, save_plots)


if __name__ == '__main__':
    if '--save-plots' in sys.argv:
        run(save_plots=False)
    else:
        run()
