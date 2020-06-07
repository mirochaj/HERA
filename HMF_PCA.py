"""
HMF_PCA.py

Author: Henri Lamarre
Affiliation: McGill University
Created on 2020-02-12.

Description: Computes a principal component analysis for 
Halo Mass Function fitting functions.
     
"""

import numpy as np
import h5py
import scipy.interpolate as sci
import matplotlib.pyplot as pl
import scipy.integrate as scig
from mpl_toolkits.mplot3d import Axes3D



def HMF_PCA(z_refined = 5, m_refined = 5, low = 142, 
            high = 450, plot = False, trunc = 30):
    ''' 
    This Function creates a PCA for 3000 HMF created with the 
    fitting functions Press-Schechter, Sheth-Mo-Tormen, Tinker. 
    The parameters 'z_refined' and 'm_refined' determine at 
    what frequency data should be discarded in order to do the PCA.
    The parameters 'high/low' gives the mass thresholds 
    at which the data is be discarded.
    The parameter 'truncate' gives the number of mass bins element 
    that should be discarded at the end of the interpolation

    -----------Parameters----------------
    z_refined: How many redshift bins are kept in the computation
    default: 1/6 bins are kept

    m_refined: How many mass bins are kept in the computation
    default: 1/5 bins are kept

    high, low: Indices of the mass bins indicating which mass
    bins are kept in the computation.
    defaults: 142-470

    plot: if the code should produce plots

    trunc: how many mass bins should be used in the computation
    but truncated in the final result
    
    ------------Creates---------------------
    An hdf5 file with the resulting eigenvectors, the mass/redshift
    bins, the coefficient needed to rebuilt fitting function HMFs.
    '''

    fitting_functions=['PS','ST','TK']
    vectors=[]
    path = '/home/henri/Documents/HERA/Cosmology/HMF_tables/'

    # The base file is just one of the HMFs from which we extract the
    # mass/redshift information which we will force to be the same 
    # for all HMFs
    base_file = h5py.File(path + 'PS/hdf5/0.hdf5','r')
    mass_complete = np.log10(list(base_file[('tab_M')]))[low:high]
    redshift_complete = list(base_file[('tab_z')])[:261]
    mass = mass_complete[::m_refined]
    redshift = redshift_complete[::z_refined]

    # For each fitting functions, load and process all the HMFs.
    for fit in fitting_functions:
        print('Processing fitting function {}'.format(fit))
        for i in range(1000):
            vectors.append(HMF_loader(i, fit, path, z_refined,
                                        m_refined, high, low))

    # Does the PCA
    e_val, e_vec, coefs, covariance = PCA(vectors)
    e_vec=e_vec[:20]

    if plot:
        plot_eigenvalues(e_val)

    ################## INTERPOLATION ######################
    '''
    The PCA was done to a lower resolution in redshift and 
    Mass for memory purposes. The HMFs are now interpolated
    back to their original resolution.
    
    The PCA does everything in 1 dimension but the HMF
    are 2D objects (Mass/Redshift) so we put them back
    in 2D arrays for the interpolation.
    '''
    e_vectors_2D = []
    for j in range(len(e_vec)):
        e_vectors_2D.append([])
        for i in range(len(redshift)):
            e_vectors_2D[j].append(e_vec[j][i*len(mass):(i+1)*len(mass)])
    
    # Creating the interpolated lists. The non-2D list is
    # going to be used for the accuracy computation
    interpolated_e_vectors_2D, interpolated_e_vectors =\
            interpolator(e_vectors_2D, mass,
                 redshift, redshift_complete, mass_complete)
    interpolated_e_vectors = np.array(interpolated_e_vectors)

    # Computes the Accuracy of the PCA
    dndm_holder = list(base_file[('tab_dndm')])[:261]
    accuracy(plot, coefs, interpolated_e_vectors,
                    dndm_holder, e_vec, high, low)

    if trunc is not 0:
        final_vectors = truncate(interpolated_e_vectors_2D, trunc)
    else:
        final_vectors = interpolated_e_vectors_2D

    # Fill the rest of the mass bins with really small numbers
    # for vec in final_vectors:
    #     for red in vec:
    #         for i in range(len(base_file[('tab_M')])-high+trunc-100):
    #             red.append(-300)



    hf = h5py.File('hmf_pca_z{}_M{}_truncated_{}_{}.hdf5'.format(z_refined, m_refined, high, low), 'w')
    hf.create_dataset('e_vec', data=final_vectors)
    hf.create_dataset('tab_z', data=redshift_complete)
    hf.create_dataset('tab_M', data=list(base_file[('tab_M')])[low:high-trunc])
    hf.create_dataset('coefs', data=coefs[:20])
    hf.close()



def Zero_replacer(vector):
    '''
    This function replaces 0 by 10e-300 in 2D arrays in order
    to avoid log problems.

    ----Parameters----
    vector: 2D array

    ----Returns----
    vector: the same array but with 0 replaced by tiny numbers
    '''
    for item in vector:
        for i in range(len(item)):
            if item[i] == 0:
                item[i] = 10**(-300)
    return vector



def HMF_loader(i, fit, path, z_refined, m_refined, high, low):
    '''
    Load the dndm information for a HMF. Discards some data.
    Stacks the 2D information of the HMF table in a 1D list
    by keeping continuity in the mass bins.

    ----Parameters----
    i: Index of the HMF table

    fit: fitting function name ('PS', 'TK' or 'ST')

    path: path to fitting function tables

    z_refined: How many redshift bins are kept in the computation
    default: 1/6 bins are kept

    m_refined: How many mass bins are kept in the computation
    default: 1/5 bins are kept

    high, low: Indices of the mass bins indicating which mass
    bins are kept in the computation.
    defaults: 142-470

    ----Returns----
    1D list of the HMF table
    '''
    f = h5py.File(path+'{}/hdf5/{}.hdf5'.format(fit,i),'r')
    dndm = np.array(list(f[('tab_dndm')])[:261][::z_refined])
    dndm = Zero_replacer(dndm)
    vector = []
    for j in range(len(dndm)):
        vector.append(np.log10(dndm[j][low:high][::m_refined]))
    if not i%100:
        print(i)
    flat = [item for sublist in vector for item in sublist]
    return flat



def plot_eigenvalues(e_val):
    '''
    Creates a plot of the eigenvalues

    ----Parameters----
    eval: the eigenvalues to plot

    '''
    pl.figure()
    pl.title('Eigenvalues')
    pl.plot(np.linspace(0,len(e_val),len(e_val)), np.log10(e_val), '.')
    pl.show()
    pl.savefig('eigenvalues.png')
    pl.close()



def interpolator(pre_interp, mass,
                 redshift, redshift_complete, mass_complete):
    '''
    Interpolates the HMFs back to the original resolution.
    Creates a sample HMF for accuracy computation

    ----Parameters----
    pre_interp: 2D HMF vector
    mass: mass array of the HMF
    redshift: redshift array of the HMF
    redshift_complete: list of redshifts to interpolate to
    mass_complete: list of masses to interpolate to

    ----Returns----
    post_interp: interpolated 2D HMF vector
    sample: HMF used in the accuracy method.
    '''

    sample = []
    post_interp = []
    print(len(mass_complete))
    for j in range(len(pre_interp)):
        sample.append([])
        post_interp.append([])
        interp = sci.interp2d(
            mass, redshift, pre_interp[j], kind = 'linear')

        for k in range(len(redshift_complete)):
            post_interp[j].append([])
            for number in mass_complete:
                sample[j].append(
                    float(interp(number, redshift_complete[k])))
                post_interp[j][k].append(
                    float(interp(number, redshift_complete[k])))
    return post_interp, sample

def accuracy(plot, coefs, vector, dndm, e_vec, high, low):
    '''
    Creates a fractionnal error accuracy test
    Prints out the percentage of points in
    the first HMF agreeing with their physical 
    counterpart up to a certain accuracy.

    ----Parameters----
    plot: if True, makes a plot of the accuracy
    coefs: coefficients needed to rebuild a fitting
    function HMF from the PCA
    vector: Sample HMF to test accuracy
    dndm: dndm from the original HMF
    e_vec: PCA vectors
    high-low: mass thresholds when loading the fitting
    function HMF
    '''
    dndm = Zero_replacer(dndm)
    synth_vec = coefs[0][0] * vector[0]
    for i in range(1,len(e_vec)):
        synth_vec += coefs[i][0] * vector[i]
    synth_vec = 10**synth_vec

    test_dndm = [item for sublist in dndm for item in sublist[low:high]]
    
    x=np.linspace(0,len(synth_vec),len(synth_vec))
    y=np.log10(np.abs(((np.array(synth_vec)-
        np.array(test_dndm))/np.array(test_dndm))))
    count = 0
    for point in y:
        if point<-2:
            count += 1
    print('Accuracy = {} Percent'.format(count/float(len(y))))
    
    if plot:
        pl.figure()
        pl.title('PCA Accuracy')
        pl.plot(x, y, '.', alpha=0.01)
        line = [-2] * len(x)
        pl.plot(x,line)
        pl.show()
        pl.savefig('Accuracy.png')
        pl.close()

def truncate(vector, trunc):
    '''
    Truncates the HMFs vectors to remove weird mgtm/ngtm
    behavior at high mass bins.

    ----Parameters----
    vector: 2D array to truncate
    trunc: number of indices to truncate
    '''
    truncated = []
    for i in range(len(vector)):
        truncated.append([])
        for j in range(len(vector[0])):
            truncated[i].append(vector[i][j][:-trunc])
    return truncated

def PCA(vectors):
    '''
    Creates a covariance matrix of an array of vectors
    Solves that covariance matrix for the eigenvectors
    and eigenvalues and return those.

    -----Parameters-----
    vectors: multi-dimensional array of vectors of same
    length

    -----Returns--------
    The eigenvectors,
    the eigenvalues,
    the coefficients needed to rebuild the original vectors
    the covariance matrix
    '''
    covariance = np.cov(np.array(vectors), rowvar = 0)
    print('Computing the {} PCA'.format(covariance.shape))
    print('Shape of the matrix is {}'.format(covariance.shape))
    e_val, e_vec = np.linalg.eigh(covariance)
    e_val=list(reversed(e_val))
    e_vec=list(reversed(e_vec.T))
    coefs = np.dot(e_vec, np.array(vectors).T)
    return e_val, e_vec, coefs, covariance