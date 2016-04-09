"""
@author: Joris van Vugt
"""

import sys
import urllib
import os
from sklearn.externals import joblib
import numpy as np

DATA_FOLDER = '/scratch/jvvugt'

NAMES = [
    'specObjID', 'targetObjID', 'z', 'zErr',

    'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
    'psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 'psfMagErr_i', 'psfMagErr_z',

    'fiberMag_u', 'fiberMag_g', 'fiberMag_r', 'fiberMag_i', 'fiberMag_z',
    'fiberMagErr_u', 'fiberMagErr_g', 'fiberMagErr_r', 'fiberMagErr_i', 'fiberMagErr_z',

    'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z',
    'modelMagErr_u', 'modelMagErr_g', 'modelMagErr_r', 'modelMagErr_i', 'modelMagErr_z',

    'petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z',
    'petroMagErr_u', 'petroMagErr_g', 'petroMagErr_r', 'petroMagErr_i', 'petroMagErr_z',

    'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z',

    'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z',

    'class'
]

DTYPES = (
    np.int64, np.int64, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,

    np.float, np.float, np.float, np.float, np.float,

    np.int
)

LABELS = ["GALAXY", "QSO", "STAR"]

def execute_query(url, query, output_format):
    """
    Encode the query in the url and execute it.
    """
    parameters = urllib.urlencode({'cmd': query, 'format': output_format})
    return urllib.urlopen(url + '?' + parameters)

def download_subsample(url, obj_class, output_format='csv'):
    """
    Download data of class obj_class from url
    """
    query = ('SELECT ' +
            ','.join(NAMES) +
            ' FROM SpecPhotoAll WHERE class="{0}" AND survey="SDSS" ORDER BY NEWID()')
    result = execute_query(url, query.format(obj_class), output_format)

    if result.getcode() != 200:
        print result.readlines()
        raise IOError('{0} - Error Fetching SDSS data' % result.getcode())

    # For some reason skyserver will just stop sending data at some point
    # That point might be in the middle of an instance, so it will have
    # less than 30 columns causing np.genfromtxt to throw an error.
    # For this reason, we use skip_footer to just disregard the last instance
    # and avoid this problem all together.
    data = np.genfromtxt(result, delimiter=',', names=NAMES,
                         dtype=DTYPES, skip_header=2,
                         skip_footer=1,
                         converters={54: lambda s: LABELS.index(s)})
    return data

def fetch_data(file_name='SDSS_DR12.p', refresh=False):
    """
    Fetch galaxy, quasar and star data from SDSS data release 12.
    The data will be saved locally to ../data, so it only has to be downloaded
    once.
    if refresh is True, then the data will always be downloaded.
    """
    path = os.path.join(DATA_FOLDER, file_name)
    if not refresh and os.path.isfile(path):
            return joblib.load(path)

    print 'Dataset not found. Downloading from server...'
    sys.stdout.flush()

    url = 'http://skyserver.sdss3.org/dr12/en/tools/search/x_sql.aspx'
    galaxies = download_subsample(url, 'GALAXY')
    stars = download_subsample(url, 'STAR')
    quasars = download_subsample(url, 'QSO')
    data = np.concatenate((galaxies, quasars, stars), axis=0)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    joblib.dump(data, path)
    return data

def pickle_csv_data(filepath, outpath):
    with open(filepath, 'r') as file:
        data = np.genfromtxt(file, delimiter=',', names=NAMES,
                         dtype=DTYPES, skip_header=1,
                         skip_footer=1,
                         converters={54: lambda s: LABELS.index(s)})
        joblib.dump(data, outpath)
        return data

def get_data_from_csv(file_name, refresh=False):
    csv_path = os.path.join(DATA_FOLDER, file_name)
    pickled_path = csv_path + '.p'
    if not refresh and os.path.isfile(pickled_path):
        return joblib.load(pickled_path)

    return pickle_csv_data(csv_path, pickled_path)



if __name__ == '__main__':
    data = fetch_data(refresh=True)
