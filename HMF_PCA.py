"""
HMF_PCA.py

Author: Henri Lamarre
Affiliation: McGill University
Created on 2020-02-12.

Description: Computes a principal component analysis for 
Halo Mass Function fitting functions.
     
"""

import time
import ares
import numpy as np
import h5py
import scipy.interpolate as sci
import matplotlib.pyplot as pl

def generate_all_hmf_tables(hmf_model='ST', hmf_path=None,
    cosmology_ids=(0, 100), cosmology_name='planck_TTTEEE_lowl_lowE',
    hmf_zmin=5, hmf_zmax=30, hmf_dz=0.05,
    hmf_logMmin=7, hmf_logMmax=17, hmf_dlogM=0.1):
    """
    Just a wrapper around `generate_hmf_table`.
    
    Will create HMF tables over range of cosmology_ids.
    """

    for i in range(*cosmology_ids):
        generate_hmf_table(hmf_model=hmf_model, hmf_path=path,
            cosmology_id=i, cosmology_name=cosmology_name,
            hmf_zmin=hmf_zmin, hmf_zmax=hmf_zmax, 
            hmf_dz=hmf_dz, hmf_logMmin=hmf_logMmin, 
            hmf_logMmax=hmf_logMmax, hmf_dlogM=hmf_dlogM)

def generate_hmf_table(hmf_model='ST', hmf_path=None,
    cosmology_id=0, cosmology_name='planck_TTTEEE_lowl_lowE',
    hmf_zmin=5, hmf_zmax=30, hmf_dz=0.05,
    hmf_logMmin=7, hmf_logMmax=17, hmf_dlogM=0.1):
    """
    Generate HMF table for given redshift/mass resolution.
    """

    kwargs = \
    {
     "hmf_path": hmf_path,
     "hmf_model": hmf_model,
     "hmf_logMmin": hmf_logMmin,
     "hmf_logMmax": hmf_logMmax,
     "hmf_dlogM": hmf_dlogM,
 
     "hmf_fmt": 'hdf5',

     # Redshift sampling
     "hmf_zmin": hmf_zmin,
     "hmf_zmax": hmf_zmax,
     "hmf_dz": hmf_dz,
 
     # Cosmology
     "cosmology_id": cosmology_id,
     "cosmology_name": cosmology_name,
     
     "progress_bar": False,
 
    }

    hmf = ares.physics.HaloMassFunction(hmf_analytic=False, 
        hmf_load=False, **kwargs)

    try:
        hmf.SaveHMF(fmt=kwargs['hmf_fmt'], clobber=False, save_MAR=False)
    except IOError as err:
        print(err)
    
def HMF_PCA(cosmology_numbers=(0, 100), cosmology_name='planck_TTTEEE_lowl_lowE',
    hmf_models=['ST'], hmf_path=None,
    hmf_zmin_pca=5, hmf_zmax_pca=30, hmf_dz_pca=0.05,
    hmf_logMmin_pca=7, hmf_logMmax_pca=17, hmf_dlogM_pca=0.1, 
    hmf_zmin_out=5, hmf_zmax_out=30, hmf_dz_out=0.05, 
    hmf_logMmin_out=7, hmf_logMmax_out=17, hmf_dlogM_out=0.01,
    pca_nmax=20, trunc=0, trim_Mmin=8., trim_Mmax=14., plot=0):
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
    
    # For each fitting functions, load and process all the HMFs.
    vectors=[]
    for fit in hmf_models:
        print('Processing fitting function {}'.format(fit))
        for _i_ in range(cosmology_numbers[0], cosmology_numbers[1]):
            #vec = HMF_loader(i, fit, path, z_refined, m_refined, high, low)
            _vec_, _hmf_, _M_ = HMF_loader(hmf_model=fit, hmf_path=hmf_path,
                cosmology_id=_i_, cosmology_name=cosmology_name,
                hmf_zmin=hmf_zmin_pca, hmf_zmax=hmf_zmax_pca, 
                hmf_dz=hmf_dz_pca, 
                hmf_logMmin=hmf_logMmin_pca, hmf_logMmax=hmf_logMmax_pca, 
                hmf_dlogM=hmf_dlogM_pca, trim_Mmin=trim_Mmin, trim_Mmax=trim_Mmax)

            vectors.append(_vec_)

    # Does the PCA
    t1 = time.time()
    e_val, e_vec, coefs, covariance = PCA(vectors)
    e_vec = e_vec[:pca_nmax]

    t2 = time.time()
    print("Done with PCA in {:.2f} seconds.".format((t2 - t1)))

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
    
    # (z, Mh) arrays used for PCA (low-res)
    mass = np.log10(_M_)
    redshift = _hmf_.tab_z
    
    hmf_logMmin_out = max(hmf_logMmin_out, trim_Mmin)
    hmf_logMmax_out = min(hmf_logMmax_out, trim_Mmax)
    
    # The arrays that we'll interpolate back to in the end.
    mass_complete = np.arange(hmf_logMmin_out, hmf_logMmax_out + hmf_dlogM_out,
        hmf_dlogM_out)
    redshift_complete = np.arange(hmf_zmin_out, hmf_zmax_out + hmf_dz_out,
        hmf_dz_out)
    
    # Interpolate back to finer mesh
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
    #dndm_holder = list(base_file[('tab_dndm')])[:261]
    #accuracy(plot, coefs, interpolated_e_vectors,
    #                dndm_holder, e_vec, high, low)

    final_vectors = interpolated_e_vectors_2D

    # Fill the rest of the mass bins with really small numbers
    # for vec in final_vectors:
    #     for red in vec:
    #         for i in range(len(base_file[('tab_M')])-high+trunc-100):
    #             red.append(-300)

    #hf = h5py.File('hmf_pca_z{}_M{}_truncated_{}_{}.hdf5'.format(z_refined, m_refined, high, low), 'w')
    
    # Add sub-str for all fitting functions
    fit_str = ''
    for fit in hmf_models:
        fit_str += fit + '_'
        
    fit_str = fit_str[0:-1]   
    fn_pca = _hmf_.tab_name.replace('hmf_{}'.format(_hmf_.pf['hmf_model']), 
        'hmf_pca_{}'.format(fit_str)) 
    
    fn_pca = fn_pca.replace('{}_'.format(str(_i_).zfill(5)), '')
    
    hf = h5py.File(fn_pca, 'w')
    hf.create_dataset('e_vec', data=final_vectors)
    hf.create_dataset('tab_z', data=redshift_complete)
    hf.create_dataset('tab_M', data=10**mass_complete)
    hf.create_dataset('coefs', data=coefs[:pca_nmax])
    hf.close()
    print('Wrote {}.'.format(fn_pca))


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



def HMF_loader(cosmology_id, cosmology_name, hmf_model='ST', hmf_path=None,
    hmf_zmin=4, hmf_zmax=30, hmf_dz=0.05, 
    hmf_logMmin=4, hmf_logMmax=18, hmf_dlogM=0.1,
    trim_Mmin=None, trim_Mmax=None):
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

    hmf = ares.physics.HaloMassFunction(hmf_model=hmf_model,
        hmf_path=hmf_path,
        cosmology_id=cosmology_id, cosmology_name=cosmology_name,
        hmf_zmin=hmf_zmin, hmf_zmax=hmf_zmax, 
        hmf_dz=hmf_dz, hmf_logMmin=hmf_logMmin,
        hmf_logMmax=hmf_logMmax, hmf_dlogM=hmf_dlogM)
    
    ok = np.ones_like(hmf.tab_M)
    if trim_Mmin is not None:
        ok[np.argwhere(hmf.tab_M < 10**trim_Mmin)] = 0
    if trim_Mmax is not None:
        ok[np.argwhere(hmf.tab_M > 10**trim_Mmax)] = 0

    dndm = hmf.tab_dndm
                    
    flat = np.log10(dndm[:,ok==1].ravel())
    
    M = hmf.tab_M[ok==1]
        
    #if not cosmology_id%100:
    #    print(cosmology_id)
    
    return flat, hmf, M

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



def interpolator(pre_interp, logmass,
                 redshift, redshift_complete, logmass_complete):
    '''
    Interpolates the HMFs back to the original resolution.
    Creates a sample HMF for accuracy computation

    ----Parameters----
    pre_interp: 2D HMF vector
    logmass: mass array of the HMF
    redshift: redshift array of the HMF
    redshift_complete: list of redshifts to interpolate to
    logmass_complete: list of masses to interpolate to

    ----Returns----
    post_interp: interpolated 2D HMF vector
    sample: HMF used in the accuracy method.
    '''

    sample = []
    post_interp = []
    print(len(logmass_complete))
    for j in range(len(pre_interp)):
        sample.append([])
        post_interp.append([])
        interp = sci.interp2d(
            logmass, redshift, pre_interp[j], kind = 'linear')

        for k in range(len(redshift_complete)):
            post_interp[j].append([])
            for number in logmass_complete:
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
    

if __name__ == '__main__':
    
    # Set this to where the HMF tables will be.
    path = '/Users/jordanmirocha/Dropbox/work/projects/henri_hmf/pr_for_pca/hmf_tables'
    
    # Uncomment this line to make a bunch of HMF tables
    # (one fitting function at a time)
    #generate_all_hmf_tables(hmf_path=path, hmf_model='ST',
    #    hmf_dz=1, hmf_dlogM=0.5, cosmology_ids=(0, 100))
    
    # Run the PCA.
    HMF_PCA(hmf_path=path, hmf_models=['ST'], hmf_dz_pca=1, hmf_dlogM_pca=0.5,
        pca_nmax=25)
    