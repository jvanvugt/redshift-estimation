"""
File for fooling around with the data
"""

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_vizier_data

def load_data():
    data = load_vizier_data('sdss_wise.tsv')
    
    X_sdss = np.empty((len(data), 5))
    X_ukidss = np.empty((len(data), 4))
    X_wise = np.empty((len(data), 4))

    X_sdss[:, 0] = data['umag']
    X_sdss[:, 1] = data['gmag']
    X_sdss[:, 2] = data['rmag']
    X_sdss[:, 3] = data['imag']
    X_sdss[:, 4] = data['zmag']

    X_ukidss[:, 0] = data['Ymag']
    X_ukidss[:, 1] = data['Jmag']
    X_ukidss[:, 2] = data['Hmag']
    X_ukidss[:, 3] = data['Kmag']
    
    X_wise[:, 0] = data['36mag']
    X_wise[:, 1] = data['45mag']

    y = data['zsp']
    
    return X_sdss, X_ukidss, X_wise, y
    
def run():
    X_sdss, X_ukidss, X_wise, y = load_data()
    missing_y = np.isnan(y)
    conf_matrix = np.zeros(7)
    for sdss, ukidss, wise, y_ in zip(X_sdss, X_ukidss, X_wise, y):
        if np.isnan(y_):
            continue
        has_sdss = not np.isnan(sdss).any()
        has_ukidss = not np.isnan(ukidss).any()
        has_wise = not np.isnan(wise).any()
        
        if has_sdss and has_ukidss and has_wise:
            conf_matrix[0] += 1
            
        elif has_sdss and has_ukidss and not has_wise:
            conf_matrix[1] += 1
            
        elif has_sdss and not has_ukidss and has_wise:
            conf_matrix[2] += 1
            
        elif has_sdss and not has_ukidss and not has_wise:
            conf_matrix[3] += 1
            
        elif not has_sdss and has_ukidss and has_wise:
            conf_matrix[4] += 1
        
        elif not has_sdss and has_ukidss and not has_wise:
            conf_matrix[5] += 1
        
        elif not has_sdss and not has_ukidss and has_wise:
            conf_matrix[6] += 1
            
        elif not has_sdss and not has_ukidss and not has_wise:
            conf_matrix[7] += 1
            
        else:
            raise ValueError('uhh wtf')
            
        
    print 'sdss, ukidss, wise'                    
    print conf_matrix
    
    print 'total present:', sum(conf_matrix)
    
    print 'total: ', len(y)
        
        
        
if __name__ == '__main__':
    run()
