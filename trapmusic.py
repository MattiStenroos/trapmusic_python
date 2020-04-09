# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:29:32 2020

This file contains functions
trapscan_presetori
trapscan_optori
@author: Matti Stenroos, matti.stenroos@aalto.fi
"""
import numpy
import scipy
def trapscan_presetori(Cmeas, Lscan, NITER):
    '''
    Performs a TRAP-MUSIC scan using sources with pre-set orientation

    Parameters
    ----------
    Cmeas : numpy.array, N_sensors x N_sensors 
        Measurement covariance matrix.
    Lscan : numpy.array, N_sensors x N_scan
        Forward model i.e. a lead field matrix that contains topographies of
        N_scan source candidates (for example, oriented normal to cortex)
    NITER : integer
        How many scanning iterations are performed; this should be equal to or
        slightly larger than the assumed number of sources.

    Returns
    -------
    maxind, mumax, mus
        maxind : indices of found sources (indices to Lscan)
        mumax  : scanning-function values for the found sources; "true"
                 sources have mumax close to 1, false sources closer to 0.
        mus    : scanning-function values for all sources all iterations.        
    
    The implementation is based on 
    Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
    MEG and EEG source localization. NeuroImage 167(2018):73--83.
    Please cite the paper, if you use the approach / function.
    
    This version is "private", for your eyes only. A public version with
    identical contents / interface will follow soon.
    v200409 (c) Matti Stenroos, matti.stenroos@aalto.fi
    '''
    Nsens = len(Lscan)
    Nscan = len(Lscan[0])
    
    # output variables
    maxind = numpy.zeros(NITER,int)
    mumax = numpy.zeros(NITER)
    mus = numpy.zeros((Nscan,NITER))
    # temporary arrays
    B = numpy.zeros((Nsens,NITER))
    Qk = numpy.zeros((Nsens,Nsens))
    
    # SVD & signal subspace of the original covariance matrix
    Utemp,_,_  = numpy.linalg.svd(Cmeas, full_matrices = True)
    Uso = Utemp[:,0:NITER]
    # the subspace basis and lead field matrix for k:th iteration
    Lthis = Lscan
    Uk = Uso

    #TRAP iteration
    for ITER in range(0,NITER):
        # subspace projection, removing the previously found topographies 
        if ITER>0:
            # apply out-projection to the forward model
            Lthis = Qk@Lscan
            Us,_,_ = numpy.linalg.svd(Qk@Uso, full_matrices = True)
            # TRAP truncation i.e. removing the previously-found topographies
            Uk = Us[:,0:(NITER-ITER+1)]
        #Norm of the current Lscan 
        Lthisnormsq = numpy.sum(Lthis*Lthis,0)
        # norm of the projection of current Lscan onto the current signal space
        PsLnormsq = numpy.sum((Uk.T@Lthis)**2,0)
        # scanning function value                    
        mus[:,ITER] = PsLnormsq/Lthisnormsq
        # maximum of the scanning function
        maxind[ITER] = numpy.argmax(mus[:,ITER])
        mumax[ITER] = mus[maxind[ITER],ITER] 
        # make the next out-projector
        if ITER<NITER-1:
            B[:,ITER] = Lscan[:,maxind[ITER]]
            l = B[:,0:(ITER+1)]
            Qk = numpy.eye(Nsens)-l@numpy.linalg.pinv(l)
    
    return [maxind,mumax,mus]

def trapscan_optori(Cmeas, Lscan, NITER, Ldim = 3):
    '''
    Performs a TRAP-MUSIC scan with optimized source orientations

    Parameters
    ----------
    Cmeas : numpy.array, N_sensors x N_sensors 
        Measurement covariance matrix.
    Lscan : numpy.array, N_sensors x (N_scan x Ldim)
        Forward model i.e. a lead field matrix that contains topographies of
        N_scan source candidates in the form [t_1i, t_1j t_1k, t_2i,...], 
        where i, j, and k mark orthogonal source orientations.
    NITER : integer
        How many scanning iterations are performed. This should be equal to or
        slightly larger than the assumed number of sources.
    Ldim : integer (optional, default 3)
        Number of (orthogonal) source components per source location.
        Typically this is 3, equaling a xyz dipole triplet, but one can also
        constrain the source space. If you want use pre-specified orientation,
        use trapscan_presetori instead.
        
    Returns
    -------
    maxind, mumax, etas. mus
        maxind : indices of found sources (indices to Lscan)
        mumax  : scanning-function values for the found sources; "true"
                 sources have mumax close to 1, false sources closer to 0.
        etamax : orientations of the found sources, (Niter x Ldim)
        mus    : scanning-function values for all sources all iterations.        
    
    The implementation is based on 
    Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
    MEG and EEG source localization. NeuroImage 167(2018):73--83.
    Please cite the paper, if you use the approach / function.
    
    This version is "private", for your eyes only. A public version with
    identical contents / interface will follow soon.
    v200409 (c) Matti Stenroos, matti.stenroos@aalto.fi
    '''
    
    Nsens, Ntopo = numpy.shape(Lscan)
    if Ntopo%Ldim:
        print('Dimensions of Lscan do not match with given Ldim.')
        return
    Nscan = int(Ntopo/Ldim)
    
    # output variables
    maxind = numpy.zeros(NITER,int)
    mumax = numpy.zeros(NITER)
    etamax = numpy.zeros((NITER,Ldim))
    mus = numpy.zeros((Nscan,NITER))
    # temporary arrays
    B = numpy.zeros((Nsens,NITER))
    Qk = numpy.zeros((Nsens,Nsens))
    
    # SVD & space of the original covariance matrix
    Utemp,_,_  = numpy.linalg.svd(Cmeas, full_matrices = True)
    Uso = Utemp[:,0:NITER]
    
    # the subspace basis and lead field matrix for k:th iteration
    Lthis = Lscan
    Uk = Uso

    # TRAP iteration
    for ITER in range(0,NITER):
        print(ITER, end = ' ')
        # subspace projection, removing previously found topographies 
        if ITER>0:
            # apply out-projection to forward model
            Lthis = Qk@Lscan
            Us,_,_ = numpy.linalg.svd(Qk@Uso, full_matrices = True)
            # TRAP truncation
            Uk = Us[:,0:(NITER-ITER)]
        
        # project L to this signal subspace
        UkLthis = Uk.T@Lthis
        # scan over all test sources
        for I in range(0,Nscan):
            # if a source has already been found for this location, skip
            if any(maxind[0:ITER] == I):
                continue
            # local lead field matrix for this source location
            L = Lthis[:,Ldim*I:(Ldim*I+Ldim)]
            UkL = UkLthis[:,Ldim*I:(Ldim*I+Ldim)]
            # find the largest mu for this L
            mus[I,ITER] = scipy.linalg.eigvalsh(UkL.T@UkL,L.T@L,eigvals = (Ldim-1,Ldim-1))
        
        # find the source with the largest mu
        mi = numpy.argmax(mus[:,ITER])
        maxind[ITER] = mi
        mumax[ITER] = mus[mi,ITER]
        
        # grab the corresponding L and extract orientation
        L = Lthis[:,Ldim*mi:(Ldim*mi+Ldim)]
        UkL = UkLthis[:,Ldim*mi:(Ldim*mi+Ldim)]
        _ ,  maxeta = scipy.linalg.eigh(UkL.T@UkL,L.T@L,eigvals = (Ldim-1,Ldim-1))
        maxeta = maxeta/scipy.linalg.norm(maxeta)   
        etamax[ITER,:] = maxeta.T
        
        # make the next out-projector
        if ITER<NITER-1:
            L = Lscan[:,Ldim*mi:(Ldim*mi+Ldim)]
            B[:,ITER] = (L@maxeta).T
            l = B[:,0:(ITER+1)]
            Qk = numpy.eye(Nsens)-l@numpy.linalg.pinv(l)
    
    print()
    return [maxind,mumax,etamax,mus]