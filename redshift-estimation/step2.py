"""
Classify high-redshift (z >= 4) and low redshift (z < 4) quasars
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix

from datasets import get_data_from_csv
from util import remove_missing

def load_data():
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

    y = np.zeros((N, 2))
    y[:, 0] = data['class']
    y[:, 1] = data['z']
    X, y = remove_missing(X, y)

    quasar_idx = y[:, 0] == 1

    X = X[quasar_idx, :]
    y = y[quasar_idx, 1]

    return X, y

def classification_info(predictions, labels, save_plots):
    """
    Print some info about the performance of the classifier.
    """
    plt.figure()
    cm = confusion_matrix(labels, predictions)
    print cm
    normalized_cm = np.log10(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    names = ['z < 4', 'z >= 4']
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Classification: normalized confusion matrix (logscale)')
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True redshift')
    plt.xlabel('Predicted redshift')

    if save_plots:
        plt.savefig('../plots/classification_z.png')
    else:
        plt.show()

    print 'accuracy:', accuracy_score(labels, predictions)
    print names
    print 'recall:', recall_score(labels, predictions, average=None)
    print 'precision:', precision_score(labels, predictions, average=None)


def run(save_plots=False):
    X, y = load_data()
    y = y >= 4
    print 'N:', len(y)
    print 'number of quasars with z >= 4:', y.sum()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    classification_info(predictions, y_test, save_plots)

if __name__ == '__main__':
    run('--save-plots' in sys.argv)