"""
PCA.py

Author: Henri Lamarre
Affiliation: McGill University
Created on 2020-02-12.

Description: Computes a principal component analysis when supplied vectors
     
"""

import numpy as np

def PCA(vectors):
    covariance = np.cov(np.array(vectors), rowvar = 0)
    print('Computing the {} PCA'.format(covariance.shape))
    print('Shape of the matrix is {}'.format(covariance.shape))
    e_val, e_vec = np.linalg.eigh(covariance)
    e_val=list(reversed(e_val))
    e_vec=list(reversed(e_vec.T))
    coefs = np.dot(e_vec, np.array(vectors).T)
    return e_val, e_vec, coefs, covariance