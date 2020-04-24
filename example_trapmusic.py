# -*- coding: utf-8 -*-
"""
This is a toy example for verifying the implementation of the 
TRAP-MUSIC algorithm (Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC
(TRAP-MUSIC) for MEG and EEG source localization. NeuroImage 167(2018):73--83. 
https://doi.org/10.1016/j.neuroimage.2017.11.013

The data simulation and scanning are in this example done using the same
source model, same source discretization, same forward model and ideal
noise. As the simulation omits all real-world errors and tests of
robustness, this script serves only as a verification of the
implementation and example for using the TRAP MUSIC. Please do not use
this kind of over-simplified simulation for any serious assessment or
method comparison.

trapmusic_python/example_trapmusic.py
trapmusic_python is licensed under BSD 3-Clause License.
Copyright (c) 2020, Matti Stenroos.
All rights reserved.
The software comes without any warranty.

v200424 Matti Stenroos, matti.stenroos@aalto.fi
"""


# Prepare a forward model:
# Make a toy forward model that has 999 sources topographies and 60 sensors.
# With default parameters, this is actually quite a difficult toy model,
# having ~7 degrees-of-freedom.

import numpy as np
from math import pi
from random import seed, sample
from trapmusic import trapscan_presetori, trapscan_optori
seed(0) 
n_sens = 60
n_sourcespace = 999
alpha = np.arange(0, n_sens)/n_sens*2*pi
L = np.zeros((n_sens, n_sourcespace))
phase_offset = np.random.rand(n_sourcespace, 1)*2*pi
omega_multip = 0.8 + np.random.rand(n_sourcespace, 1)*0.4
for i in range(0, n_sourcespace):
    L[:, i] = np.sin(omega_multip[i]*(phase_offset[i]+alpha))

# Check how nasty the model turned out
_,s,_ = np.linalg.svd(L@L.T);
relcond = s[0]/s;
dof = np.where(relcond > 1e6)[0][0]
print('The forward model has approx.', dof + 1, 'degrees of freedom.');

#%%
# Simulation 1, pre-set source orientation-
# Assume that each topography (column) in L represents a source point with
# (assumed) known orientation. In this case, each source location has 
# only one possible topography, corresponding to "fixed orientation"
# in EEG/MEG language.

n_truesources = 5 #how many sources there are
n_iter = n_truesources + 2 #how many sources we guess there to be
n_rep = 10 #how many different runs are done
pSNR = 1  #power-SNR, pSNR=trace(C_idealmeas)/trace(C_noise)
C_source = 0.2*np.ones((n_truesources, n_truesources)) + 0.8*np.eye(n_truesources) #source covariance
seed(0)
print('Simulation: pSNR', pSNR, ',', n_truesources,'sources,', n_rep,'runs, pre-set orientation');
for i in range(0, n_rep):
    # Make a multi-source set
    sourceinds_true = sample(range(0, n_sourcespace), n_truesources)
    # Generate measurement covariance matrix
    L_this = L[:, sourceinds_true]
    C0 = L_this@C_source@L_this.T
    C_noise = np.trace(C0)/(pSNR*n_sens)*np.eye(n_sens)
    # pSNRtest = trace(C0)/trace(C_noise);
    C_meas = C0 + C_noise
    #do a TRAP scan with pre-set orientations
    sourceinds_trap, mu_max,_ = trapscan_presetori(C_meas, L, n_iter)     
    #check how it went
    ind_found = np.isin(sourceinds_trap, sourceinds_true)
    print("{0:2d}: found {1:1d}/{2:1d} sources".format(i, sum(ind_found), n_truesources), end = "")
    if any(ind_found):
        mu_truemin = np.min(mu_max[ind_found])
        print(", min(mu_true) = {0:.3f}".format(mu_truemin), end = "")
    if any(~ind_found):
        mu_falsemax = np.max(mu_max[~ind_found])
        print(", max(mu_false)= {0:.3f}".format(mu_falsemax))
    else:
        print()
        
#%%
# Simulation 2, unknown / optimized orientation.
# Now, assume that L has 333 source locations and each source location has
# three possible (orthogonal) topographies. For L, this situation
# corresponds to "free orientation" or "vector source". The TRAP MUSIC
# algorithm assumes that each source location has a constant unknown
# orientation, and searches for the orientation that most strongly projects to 
# the signal space. This corresponds to the typical formulations with
# optimal-orientation scalar beamformers.

n_truesources = 5 #how many sources there are
n_iter = n_truesources + 2 #how many sources we guess there to be
n_rep = 10 #how many different runs are done
pSNR = 1 #power-SNR, pSNR=trace(C_idealmeas)/trace(C_noise)
C_source = 0.2*np.ones((n_truesources, n_truesources)) + 0.8*np.eye(n_truesources) #source covariance

n_sourcepos = n_sourcespace//3
np.random.seed(0)
print()
print("Simulation: pSNR {0:.1f}, {1:1d} sources, {2:2d} runs, optimized orientation."
      .format(pSNR,n_truesources,n_rep))
for i in range(0, n_rep):
    # Make a multi-source set
    sourceinds_true = sample(range(0, n_sourcepos), n_truesources)
    oritemp = np.random.rand(n_truesources, 3) - .5
    orinorm = np.sqrt(np.sum(oritemp**2,1))
    sourceoris_true = np.diag(1/orinorm)@oritemp 
    # Extract oriented sources & make forward mapping for them
    L_this = np.zeros((n_sens, n_truesources))
    for j in range(0, n_truesources):
        L_local = L[:, 3*sourceinds_true[j] + np.array([0, 1, 2])]
        L_this[:, j] = L_local@sourceoris_true[j, :].T
    # Generate measurement covariance matrix
    C0 = L_this@C_source@L_this.T
    C_noise = np.trace(C0)/(pSNR*n_sens)*np.eye(n_sens)
    # pSNRtest = trace(C0)/trace(C_noise);
    C_meas = C0 + C_noise
    #do a TRAP scan with optimized orientations
    sourceinds_trap, mu_max, eta_mumax,_ = trapscan_optori(C_meas, L, n_iter)
    
    #check how it went
    ind_found = np.isin(sourceinds_trap, sourceinds_true)
    #check how it went
    ind_found = np.isin(sourceinds_trap, sourceinds_true)
    print("{0:2d}: found {1:1d}/{2:1d} sources".format(i, sum(ind_found), n_truesources), end = "")
    if any(ind_found):
        mu_truemin = np.min(mu_max[ind_found])
        print(", min(mu_true) = {0:.3f}".format(mu_truemin), end = "")
    if any(~ind_found):
        mu_falsemax = np.max(mu_max[~ind_found])
        print(", max(mu_false)= {0:.3f}".format(mu_falsemax))
    else:
        print()
    if any(ind_found):
        ind_match, ia, ib = np.intersect1d(sourceinds_true,sourceinds_trap, return_indices=True)
        oris_found = eta_mumax[ib,:];
        oris_found = np.diag(1/np.sqrt(np.sum(oris_found**2, 1)))@oris_found
        oris_ref = sourceoris_true[ia,:];
        oris_ref = np.diag(1/np.sqrt(np.sum(oris_ref**2, 1)))@oris_ref              
        oris_diff = 180/pi*np.arccos(np.round(np.abs(np.sum(oris_ref*oris_found, 1)), 6))
        print('    orientation errors: min {0:.2f} deg, max {1:.2f} deg.'
          .format(np.min(oris_diff), np.max(oris_diff))) 