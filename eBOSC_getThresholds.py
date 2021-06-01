# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:02:43 2021

@author: Lemon
"""

#    This file is part of the extended Better OSCillation detection (eBOSC) library.
#
#    The eBOSC library is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    The eBOSC library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2020 Julian Q. Kosciessa, Thomas H. Grandy, Douglas D. Garrett & Markus Werkle-Bergner

import numpy as np

def eBOSC_getThresholds(cfg,TFR,eBOSC):

# This function estimates the static duration and power thresholds and
# saves information regarding the overall spectrum and background.
#
# Inputs: 
#           cfg | config structure with cfg.eBOSC field
#           data | time-frequency matrix in the shape of (trial,freq,timepoints). One channel only
#           eBOSC | main eBOSC output structure; will be updated
#
# Outputs: 
#           eBOSC   | updated w.r.t. background info (see below)
#                   | bg_pow: overall power spectrum
#                   | bg_log10_pow: overall power spectrum (log10)
#                   | pv: intercept and slope of fit
#                   | mp: linear background power
#                   | pt: power threshold
#           pt | empirical power threshold
#           dt | duration threshold

    trial_background = cfg['trial_background']
    background_sample = cfg['background_sample']
    total_sample = cfg['total_sample']
    excludePeak = cfg['excludePeak']
    wavenumber = cfg['wavenumber']
    threshold_pct = cfg['threshold_pct']
    threshold_duration = cfg['threshold_duration']
    fsample = cfg['fsample']
    F = cfg['F']

# average power estimates across periods of interest
    BG=[]
    dummy = TFR[trial_background,:,background_sample:-1-background_sample]
    BG = dummy.reshape(dummy.shape[0]*dummy.shape[2],dummy.shape[1])
    BG = np.transpose(BG)
    del dummy
    
    # if frequency ranges should be exluded to reduce the influence of
    # rhythmic peaks on the estimation of the linear background, the
    # following section removes these specified ranges
    freqKeep = np.array([True] *len(F))
    if excludePeak: # allow for no peak removal
        for exFreq in excludePeak: # allow for multiple peaks
            # find empirical peak in specified range
            freqInd1 = np.where(F >= exFreq[0])[0][0]
            freqInd2 = np.where(F <= exFreq[1])[0][-1]
            freqidx = list(range(freqInd1,freqInd2+1))
            indPos = np.argmax(np.mean(BG[freqidx,:],1))
            indPos = freqidx[indPos]
            # approximate wavelet extension in frequency domain
            # note: we do not remove the specified range, but the FWHM
            # around the empirical peak
            LowFreq = F[indPos]-(((2/wavenumber)*F[indPos])/2)
            UpFreq = F[indPos]+ (((2/wavenumber)*F[indPos])/2)
            # index power estimates within the above range to remove from BG fit
            freqKeep[ (F >= LowFreq) & (F <= UpFreq)] = 0

    fitInput  = { 'f_': F[freqKeep] } 
    fitInput['BG_'] = BG[freqKeep, :];
        
    # robust linear regression
    import statsmodels.api as sm
    rlm_model = sm.RLM(np.mean(np.log10(fitInput['BG_']),axis = 1),np.log10(fitInput['f_']), M=sm.robust.norms.TukeyBiweight())

    rlm_results = rlm_model.fit()


    # perform the robust linear fit, only including putatively aperiodic components (i.e., peak exclusion)
    b = rlm_results.params
    del fitInput
    
    pv = []
    pv[0] = b[1]
    pv[1] = b[0]
    mp = 10**rlm_results.fittedvalues

    # compute eBOSC power (pt) and duration (dt) thresholds: 
    # power threshold is based on a chi-square distribution with df=2 and mean as estimated above
    from scipy.stats.distributions import chi2

    pt=chi2.ppf(threshold_pct,2)*mp/2 # chi2inv.m is part of the statistics toolbox of Matlab and Octave
    # duration threshold is the specified number of cycles, so it scales with frequency
    dt=threshold_duration*fsample/F;

    eBOSC = {}
    # save multiple time-invariant estimates that could be of interest:
    # overall wavelet power spectrum (NOT only background)
    eBOSC['bg_pow']        = np.mean(BG[:,total_sample:(-1-total_sample)],1);
    # log10-transformed wavelet power spectrum (NOT only background)
    eBOSC['bg_log10_pow'] = np.mean(np.log10(BG[:,total_sample:(-1-total_sample)]),1);
    # intercept and slope parameters of the robust linear 1/f fit (log-log)
    eBOSC['pv']            = pv;
    # linear background power at each estimated frequency
    eBOSC['mp']          = mp;
    # statistical power threshold
    eBOSC['pt']           = pt;

    return eBOSC, pt, dt
