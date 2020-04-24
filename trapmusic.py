# -*- coding: utf-8 -*-
"""
trapmusic_python
trapmusic_python is licensed under BSD 3-Clause License.
Copyright (c) 2020, Matti Stenroos.
All rights reserved.
The software comes without any warranty.
"""
import numpy as np
from scipy.linalg import eigh, eigvalsh
def trapscan_presetori(C_meas, L_scan, n_iter):
    '''
    Performs a TRAP-MUSIC scan using sources with pre-set orientation

    Parameters
    ----------
    C_meas : numpy.array, n_sensors x n_sensors 
        Measurement covariance matrix.
    L_scan : numpy.array, n_sensors x n_scan
        Forward model i.e. a lead field matrix that contains topographies of
        n_scan source candidates (for example, oriented normal to cortex)
    n_iter : integer
        How many scanning iterations are performed; this should be equal to or
        slightly larger than the assumed number of sources.

    Returns
    -------
    ind_max, mu_max, mus
        ind_max : indices of found sources (indices to L_scan)
        mu_max  : scanning-function values for the found sources; "true"
                 sources have mu_max close to 1, false sources closer to 0.
        mus    : scanning-function values for all sources all iterations.        
    
    Based on  
    Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
    MEG and EEG source localization. NeuroImage 167(2018):73--83.
    https://doi.org/10.1016/j.neuroimage.2017.11.013
    For further information, please see the paper. I also kindly ask you to 
    cite the paper, if you use the approach and/or this implementation.
    If you do not have access to the paper, please send a request by email.
    
    trapmusic_python.trapscan_presetori
    trapmusic_python is licensed under BSD 3-Clause License.
    Copyright (c) 2020, Matti Stenroos.
    All rights reserved.
    The software comes without any warranty.
    
    v200424 Matti Stenroos, matti.stenroos@aalto.fi
    '''
    n_sens, n_scan = np.shape(L_scan)
    # output variables
    ind_max = np.zeros(n_iter, int)
    mu_max = np.zeros(n_iter)
    mus = np.zeros((n_scan, n_iter))
    # temporary arrays
    B = np.zeros((n_sens, n_iter))
    Qk = np.zeros((n_sens, n_sens))
    
    # SVD & signal subspace of the original covariance matrix
    Utemp,_,_  = np.linalg.svd(C_meas, full_matrices = True)
    Uso = Utemp[:, 0:n_iter]
    # the subspace basis and lead field matrix for k:th iteration
    L_this = L_scan
    Uk = Uso

    #TRAP iteration
    for iter in range(0, n_iter):
        # subspace projection, removing the previously found topographies 
        if iter > 0:
            # apply out-projection to the forward model
            L_this = Qk@L_scan
            Us,_,_ = np.linalg.svd(Qk@Uso, full_matrices = True)
            # TRAP truncation i.e. removing the previously-found topographies
            Uk = Us[:, 0:(n_iter - iter + 1)]
        #Norm of the current L_scan 
        L_thisnormsq = np.sum(L_this*L_this, 0)
        # norm of the projection of current L_scan onto the current signal space
        PsLnormsq = np.sum((Uk.T@L_this)**2, 0)
        # scanning function value                    
        mus[:, iter] = PsLnormsq/L_thisnormsq
        # with poorly-visible sources, numerical behavior might lead to
        # re-finding the same source again (despite out-projection) -> remove
        if iter > 0:
            mus[ind_max[0:iter],iter] = 0
        # maximum of the scanning function
        ind_max[iter] = np.argmax(mus[:, iter])
        mu_max[iter] = mus[ind_max[iter], iter] 
        # make the next out-projector
        if iter < n_iter - 1:
            B[:, iter] = L_scan[:, ind_max[iter]]
            l = B[:, 0:(iter + 1)]
            Qk = np.eye(n_sens) - l@np.linalg.pinv(l)
    
    return ind_max, mu_max, mus

def trapscan_optori(C_meas, L_scan, n_iter, Ldim = 3):
    '''
    Performs a TRAP-MUSIC scan with optimized source orientations

    Parameters
    ----------
    C_meas : numpy.array, N_sensors x N_sensors 
        Measurement covariance matrix.
    L_scan : numpy.array, N_sensors x (N_scan x Ldim)
        Forward model i.e. a lead field matrix that contains topographies of
        N_scan source candidates in the form [t_1i, t_1j t_1k, t_2i,...], 
        where i, j, and k mark orthogonal source orientations.
    n_iter : integer
        How many scanning iterations are performed. This should be equal to or
        slightly larger than the assumed number of sources.
    Ldim : integer (optional, default 3)
        Number of (orthogonal) source components per source location.
        Typically this is 3, equaling a xyz dipole triplet, but one can also
        constrain the source space. If you want use pre-specified orientation,
        use trapscan_presetori instead.
        
    Returns
    -------
    ind_max, mu_max, etas,  mus
        ind_max : indices of found sources (indices to L_scan)
        mu_max  : scanning-function values for the found sources; "true"
                 sources have mu_max close to 1, false sources closer to 0.
        eta_max : orientations of the found sources, (n_iter x Ldim)
        mus    : scanning-function values for all sources all iterations.        
    
    Based on  
    Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
    MEG and EEG source localization. NeuroImage 167(2018):73--83.
    https://doi.org/10.1016/j.neuroimage.2017.11.013
    For further information, please see the paper. I also kindly ask you to 
    cite the paper, if you use the approach and/or this implementation.
    If you do not have access to the paper, please send a request by email.
    
    trapmusic_python.trapscan_optori
    trapmusic_python is licensed under BSD 3-Clause License.
    Copyright (c) 2020, Matti Stenroos.
    All rights reserved.
    The software comes without any warranty.
    
    v200424 Matti Stenroos, matti.stenroos@aalto.fi
    '''
    
    n_sens, Ntopo = np.shape(L_scan)
    if Ntopo%Ldim:
        raise ValueError('Dimensions of L_scan do not match with given Ldim.')
        return
    n_scan = int(Ntopo/Ldim)
    
    # output variables
    ind_max = np.zeros(n_iter, int)
    mu_max = np.zeros(n_iter)
    eta_max = np.zeros((n_iter, Ldim))
    mus = np.zeros((n_scan, n_iter))
    # temporary arrays
    B = np.zeros((n_sens, n_iter))
    Qk = np.zeros((n_sens, n_sens))
    
    # SVD & space of the original covariance matrix
    Utemp,_,_  = np.linalg.svd(C_meas, full_matrices = True)
    Uso = Utemp[:, 0:n_iter]
    
    # the subspace basis and lead field matrix for k:th iteration
    L_this = L_scan
    Uk = Uso

    # TRAP iteration
    for iter in range(0, n_iter):
        # subspace projection, removing previously found topographies 
        if iter>0:
            # apply out-projection to forward model
            L_this = Qk@L_scan
            Us,_,_ = np.linalg.svd(Qk@Uso, full_matrices = True)
            # TRAP truncation
            Uk = Us[:, 0:(n_iter - iter)]
        
        # project L to this signal subspace
        UkL_this = Uk.T@L_this
        # scan over all test sources
        for i in range(0, n_scan):
            # if a source has already been found for this location, skip
            if any(ind_max[0:iter] == i):
                continue
            # local lead field matrix for this source location
            L = L_this[:, Ldim*i:(Ldim*i + Ldim)]
            UkL = UkL_this[:, Ldim*i:(Ldim*i + Ldim)]
            # find the largest mu for this L
            mus[i, iter] = eigvalsh(UkL.T@UkL, L.T@L, eigvals = (Ldim - 1, Ldim - 1))
        
        # find the source with the largest mu
        mi = np.argmax(mus[:, iter])
        ind_max[iter] = mi
        mu_max[iter] = mus[mi, iter]
        
        # grab the corresponding L and extract orientation
        L = L_this[:, Ldim*mi:(Ldim*mi + Ldim)]
        UkL = UkL_this[:, Ldim*mi:(Ldim*mi + Ldim)]
        _ ,  maxeta = eigh(UkL.T@UkL, L.T@L, eigvals = (Ldim - 1, Ldim - 1))
        maxeta = maxeta/np.linalg.norm(maxeta)   
        eta_max[iter, :] = maxeta.T
        
        # make the next out-projector
        if iter < n_iter - 1:
            L = L_scan[:, Ldim*mi:(Ldim*mi + Ldim)]
            B[:, iter] = (L@maxeta).T
            l = B[:, 0:(iter + 1)]
            Qk = np.eye(n_sens) - l@np.linalg.pinv(l)
    
    return ind_max, mu_max, eta_max, mus