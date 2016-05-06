import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from fetch_sdss_data import load_vizier_data
from pipeline import remove_missing, regressor_info

"""
1. Load data
2. Remove entries that are missing y or all of X
3. Train regressors on X_1 and X_2
"""

def get_data():
    data = load_vizier_data('sdss_wise.tsv')
    X_sdss = np.empty((len(data), 5))
    X_wise = np.empty((len(data), 4))

    X_sdss[:, 0] = data['umag']
    X_sdss[:, 1] = data['gmag']
    X_sdss[:, 2] = data['rmag']
    X_sdss[:, 3] = data['imag']
    X_sdss[:, 4] = data['zmag']

    X_wise[:, 0] = data['Ymag']
    X_wise[:, 1] = data['Jmag']
    X_wise[:, 2] = data['Hmag']
    X_wise[:, 3] = data['Kmag']

    y = data['zsp']

    return X_sdss, X_wise, y

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


def run():
    X_sdss, X_wise, y = get_data()
    X = np.hstack((X_sdss, X_wise))
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), random_state=42)
    X_train_sdss = X_train[:, :5]
    X_train_wise = X_train[:, 5:]

    X_train_sdss, y_train_sdss = remove_missing(X_train_sdss, y_train)
    X_train_wise, y_train_wise = remove_missing(X_train_wise, y_train)

    y_train_sdss.shape = (-1,)
    y_train_wise.shape = (-1,)
    sdss_reg = RandomForestRegressor(n_estimators=100, n_jobs=-2)
    sdss_reg.fit(X_train_sdss, y_train_sdss)

    wise_reg = RandomForestRegressor(n_estimators=100, n_jobs=-2)
    wise_reg.fit(X_train_wise, y_train_wise)

    X_test_sdss_idx = ~np.isnan(X_test[:, :5]).any(axis=1)
    X_test_wise_idx = ~np.isnan(X_test[:, 5:]).any(axis=1)

    pred_sdss = sdss_reg.predict(X_test[X_test_sdss_idx, :5])
    pred_wise = wise_reg.predict(X_test[X_test_wise_idx, 5:])
    
    print len(X_test_sdss_idx)
    
    not_missing_idx = np.logical_or(X_test_sdss_idx, X_test_wise_idx)
    
    actual = y_test[not_missing_idx]
    N = len(actual)
    print 'test samples: ', N
    predictions = np.full(.0N, float('NaN'), dtype=np.float32)
    predictions[X_test_sdss_idx] = pred_sdss

    for i, use in enumerate(X_test_wise_idx):
        if not use:
            continue
        if not np.isnan(predictions[i]):
            predictions[i] = (predictions[i] + pred_wise[i]) / 2.
        else:
            predictions[i] = pred_wise[i]

    # not_nan = ~np.isnan(predictions)
    # predictions = predictions[not_nan]
    # actual = actual[not_nan]

    regressor_info(predictions.reshape(-1,), actual.reshape(-1,), True)



if __name__ == '__main__':
    run()
