"""
@author: Joris van Vugt
"""

import sys
import urllib
import os
from sklearn.externals import joblib
import numpy as np

DATA_FOLDER = '../data'

NAMES = [
    'specObjID', 'targetObjID', 'z', 'zErr', 'psfMag_u', 'psfMagErr_u',
    'psfMag_g', 'psfMagErr_g', 'psfMag_r', 'psfMagErr_r', 'psfMag_i',
    'psfMagErr_i', 'psfMag_z', 'psfMagErr_z', 'modelMag_u', 'modelMagErr_u',
    'modelMag_g', 'modelMagErr_g', 'modelMag_r', 'modelMagErr_r', 'modelMag_i',
    'modelMagErr_i', 'modelMag_z', 'modelMagErr_z', 'dered_u', 'dered_g',
    'dered_r', 'dered_i', 'dered_z', 'class'
]

DTYPES = (
    np.int64, np.int64,
    np.float, np.float, np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float, np.float, np.float,
    np.float, np.float, np.float, np.float, np.float, np.float,
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
                         converters={29: lambda s: LABELS.index(s)})
    return data

def fetch_data(refresh=False):
    """
    Fetch galaxy, quasar and star data from SDSS data release 10.
    The data will be saved locally to ../data, so it only has to be downloaded
    once.
    if refresh is True, then the data will always be downloaded.
    """
    file_name = 'SDSS_DR10.p'
    path = os.path.join(DATA_FOLDER, file_name)
    if not refresh and os.path.isfile(path):
            return joblib.load(path)

    print 'Dataset not found. Downloading from server...'
    sys.stdout.flush()

    url = 'http://skyserver.sdss3.org/dr10/en/tools/search/x_sql.aspx'
    galaxies = download_subsample(url, 'GALAXY')
    stars = download_subsample(url, 'STAR')
    quasars = download_subsample(url, 'QSO')
    data = np.concatenate((galaxies, quasars, stars), axis=0)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    joblib.dump(data, path)
    return data



if __name__ == '__main__':
    data = fetch_data(refresh=True)
