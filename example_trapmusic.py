# -*- coding: utf-8 -*-
"""
This script demonstrates the use of TRAP-MUSIC algorithm.

The TRAP-MUSIC implementation is based on 
    Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
    MEG and EEG source localization. NeuroImage 167(2018):73--83.
Please cite the paper, if you use the approach / function.
@author: Matti Stenroos
v200408
"""
#!python
#!/usr/bin/env python
import scipy.io
import numpy
import trapmusic

# load example data that contains 'Cmeas','Lxyz' and 'Ln' 
modelfile = 'd:/tmp/TRAPtestmodel_v7.mat'
datastruct = scipy.io.loadmat(modelfile)

# measurement covariance matrix, (number of sensors x number of sensors)
Cmeas = numpy.array(datastruct['Cmeas'])

# a forward model that has known source orientation for each source candidate,
# (number of sensors x number of source candidates)
Lscan_presetori = numpy.array(datastruct['Ln'])
# a forward model that x, y, and z orientations for each source candidate,
# (number of sensors x (3 x number of source candidates))
Lscan_xyzori = numpy.array(datastruct['Lxyz'])

# how many "true" sources we assume i.e. how many sources are sought
NITER = 6

# scan using pre-set orientations
mi_ps,mv_ps,muu_ps = trapmusic.trapscan_presetori(Cmeas,Lscan_presetori,NITER)
# scan using optimized orientation
mi_oo,mv_oo,etas,muu_oo = trapmusic.trapscan_optori(Cmeas,Lscan_xyzori,NITER)
