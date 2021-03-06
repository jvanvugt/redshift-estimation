"""
Redshift estimation for quasars
"""

import sys
from math import sqrt, floor, ceil

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import *

from datasets import load_vizier_data
from util import remove_missing

def get_data():
    data = load_vizier_data('sdss_wise.tsv')
    X_sdss_wise = np.empty((len(data), 7))
    X_ukidss = np.empty((len(data), 4))

    X_sdss_wise[:, 0] = data['umag']
    X_sdss_wise[:, 1] = data['gmag']
    X_sdss_wise[:, 2] = data['rmag']
    X_sdss_wise[:, 3] = data['imag']
    X_sdss_wise[:, 4] = data['zmag']
    X_sdss_wise[:, 5] = data['36mag']
    X_sdss_wise[:, 6] = data['45mag']

    X_ukidss[:, 0] = data['Ymag']
    X_ukidss[:, 1] = data['Jmag']
    X_ukidss[:, 2] = data['Hmag']
    X_ukidss[:, 3] = data['Kmag']

    y = data['zsp']

    return X_sdss_wise, X_ukidss, y

def plot_importances(forest, N_features):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(N_features):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(N_features), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(N_features), indices)
    plt.xlim([-1, N_features])
    plt.show()

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
        plt.savefig('../plots/regressor.png')
    else:
        plt.show()
        
def complete_data(save_plots=False):
    X_sdss_wise, X_ukidss, y = get_data()
    X = np.hstack((X_sdss_wise, X_ukidss))

    high_z_idx = y >= 4
    X = X[high_z_idx, :]
    y = y[high_z_idx]
    print X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)
    
    clf_sdss_wise = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    clf_all = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    
    clf_sdss_wise.fit(X_train[:, :7], y_train)
    
    X_train_all, y_train_all = remove_missing(X_train, y_train)
    
    clf_all.fit(X_train_all, y_train_all)
    
    non_missing_idx = ~np.isnan(X_test).any(axis=1)
    predictions = clf_sdss_wise.predict(X_test[:, :7])
    predictions[non_missing_idx] = clf_all.predict(X_test[non_missing_idx, :])
    
    regressor_info(predictions, y_test, save_plots)
    
    


def run(save_plots=False):
    X_sdss_wise, X_ukidss, y = get_data()

    # Combine the datasets, so we can split the training and test sets
    imputer = Imputer(strategy='median', axis=1)
    X = np.hstack((X_sdss_wise, X_ukidss))
    print 'Initial dataset size: ', len(X)
    print 'N rows without missing values: ', sum(~np.isnan(X).any(axis=1))
    X = imputer.fit_transform(X)
    print 'N rows without missing values after: ', sum(~np.isnan(X).any(axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Split the two features so we can train seperate models on them
    X_train_sdss = X_train[:, :5]
    X_train_ukidss = X_train[:, 5:]

    # Remove training data with missing values
    X_train_sdss, y_train_sdss = remove_missing(X_train_sdss, y_train)
    X_train_ukidss, y_train_ukidss = remove_missing(X_train_ukidss, y_train)

    print 'N_train_sdss: ', len(X_train_sdss)
    print 'N_train_ukidss: ', len(X_train_ukidss)

    # Train a model for both features
    print 'Training models...'
    sdss_reg = RandomForestRegressor(n_estimators=100, n_jobs=-2)
    sdss_reg.fit(X_train_sdss, y_train_sdss)

    ukidss_reg = RandomForestRegressor(n_estimators=100, n_jobs=-2)
    ukidss_reg.fit(X_train_ukidss, y_train_ukidss)

    # Find the indices of rows that aren't missing
    X_test_sdss_idx = ~np.isnan(X_test[:, :5]).any(axis=1)
    X_test_ukidss_idx = ~np.isnan(X_test[:, 5:]).any(axis=1)

    # Train the models
    print 'Calculating predictions...'
    pred_sdss = np.full(len(X_test), np.nan, dtype=np.float32)
    pred_sdss[X_test_sdss_idx] = sdss_reg.predict(X_test[X_test_sdss_idx, :5])
    print 'test set sdss size: ', sum(X_test_sdss_idx)
    pred_ukidss = np.full(len(X_test), np.nan, dtype=np.float32)
    print 'test set ukidss size: ', sum(X_test_ukidss_idx)
    pred_ukidss[X_test_ukidss_idx] = ukidss_reg.predict(X_test[X_test_ukidss_idx, 5:])

    predictions = np.full(len(X_test), np.nan, dtype=np.float32)
    for i, (sdss, ukidss) in enumerate(zip(pred_sdss, pred_ukidss)):
        if not np.isnan(sdss) and not np.isnan(ukidss):
            predictions[i] = (sdss + ukidss) / 2.
        elif not np.isnan(sdss):
            pass
            predictions[i] = sdss
        elif not np.isnan(ukidss):
            predictions[i] = ukidss

    print 'Original size test set: ', len(y_test)
    pred_idx = ~np.isnan(predictions)
    predictions = predictions[pred_idx]
    actual = y_test[pred_idx]

    print 'New size test set: ', len(predictions)


    regressor_info(predictions, actual, save_plots)



if __name__ == '__main__':
    complete_data('--save-plots' in sys.argv)
