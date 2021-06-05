# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:11:30 2021

@author: Lemon
"""

#    This file is part of the Better OSCillation detection (BOSC) library.
#
#    The BOSC library is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    The BOSC library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2010 Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten
#    and Clayton T. Dickson.
import numpy as np

def tf(eegsignal,F,Fsample,wavenumber):
# [B,T,F]=BOSC_tf(eegsignal,F,Fsample,wavenumber);
#
# This function computes a continuous wavelet (Morlet) transform on
# a segment of EEG signal; this can be used to estimate the
# background spectrum (BOSC_bgfit) or to apply the BOSC method to
# detect oscillatory episodes in signal of interest (BOSC_detect).
#
# parameters:
# eegsignal - a row vector containing a segment of EEG signal to be
#             transformed
# F - a set of frequencies to sample (Hz)
# Fsample - sampling rate of the time-domain signal (Hz)
# wavenumber is the size of the wavelet (typically, width=6)
#	
# returns:
# B - time-frequency spectrogram: power as a function of frequency
#     (rows) and time (columns)
# T - vector of time values (based on sampling rate, Fsample)

    st=1./(2*np.pi*(F/wavenumber))
    A=1./np.sqrt(st*np.sqrt(np.pi))
    B = np.zeros((len(F),len(eegsignal))) # initialize the time-frequency matrix
    for f in range(len(F)): # loop through sampled frequencies
      t=np.arange(-3.6*st[f],3.6*st[f], step = 1/Fsample)
      m=A[f]*np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*F[f]*t) # Morlet wavelet
      y=np.convolve(eegsignal,m)
      y=abs(y)**2
      B[f,:]=y[np.ceil(len(m)/2).astype(int): (len(y)-np.floor(len(m)/2).astype(int))+1]
    #T=(1:size(eegsignal,2))/Fsample;

    return B


def detect(b,powthresh,durthresh,Fsample):
# detected=BOSC_detect(b,powthresh,durthresh,Fsample)
#
# This function detects oscillations based on a wavelet power
# timecourse, b, a power threshold (powthresh) and duration
# threshold (durthresh) returned from BOSC_thresholds.m.
#
# It now returns the detected vector which is already episode-detected.
#
# b - the power timecourse (at one frequency of interest)
#
# durthresh - duration threshold in  required to be deemed oscillatory
# powthresh - power threshold
#
# returns:
# detected - a binary vector containing the value 1 for times at
#            which oscillations (at the frequency of interest) were
#            detected and 0 where no oscillations were detected.
# 
# NOTE: Remember to account for edge effects by including
# "shoulder" data and accounting for it afterwards!
#
# To calculate Pepisode:
# Pepisode=length(find(detected))/(length(detected));                           

    #t=np.arange(1,len(b)+1)/Fsample;
    nT=len(b); # number of time points
    
    x=(b>powthresh).astype(int) # Step 1: power threshold
    dx=np.diff(x)
    pos=list(np.where(dx==1)[0]+1)
    neg=list(np.where(dx==-1)[0]+1) # show the +1 and -1 edges
    
    # now do all the special cases to handle the edges
    
    if len(pos)==0 and len(neg) ==0:
      if all(x):
          H = np.asarray(([1],[nT]))
      else:
          H=[] # all episode or none
    elif len(pos)==0:
        H = np.asarray(([1],[neg])) # i.e., starts on an episode, then stops
    elif len(neg)==0:
        H = np.asarray(([pos],[nT])) # starts, then ends on an ep.
    else:
      if pos[0]>neg[0]:
          pos=[1] + pos # we start with an episode
      if neg[-1]<pos[-1]:
          neg=neg + [nT] # we end with an episode
      H = np.asarray((pos,neg)) # NOTE: by this time, length(pos)==length(neg), necessarily
    # special-casing, making the H double-vector
    
    if H.size != 0: # more than one "hole"
                    # find epochs lasting longer than minNcycles*period
      goodep=list(np.where((H[1]-H[0])>=durthresh)[0])
      if len(goodep)==0:
          H=[]
      else:
          H=H[:,goodep] 
      # OR this onto the detected vector
      detected=np.zeros(b.shape)
      for h in range(H.shape[1]):
          detected[H[0,h]:H[1,h]]=1
    # more than one "hole"

    return detected

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
def getThresholds(cfg,TFR,eBOSC):

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
    excludePeak = cfg['excludePeak'] # e.g.excludePeak = np.asarray(([2,8]))
    wavenumber = cfg['wavenumber']
    threshold_pct = cfg['threshold_pct']
    threshold_duration = cfg['threshold_duration']
    fsample = cfg['fsample']
    F = cfg['F']
    
    if len(TFR.shape) <3:
        TFR = TFR[np.newaxis]
        
    if len(excludePeak.shape) <2:
        excludePeak = excludePeak[np.newaxis]
        
    if trial_background == 'all':
        trial_background = range(TFR.shape[0])

# average power estimates across periods of interest
    BG = TFR[trial_background,:,background_sample:TFR.shape[2]-background_sample+1]
    BG = np.mean(BG,0)
    
    # if frequency ranges should be exluded to reduce the influence of
    # rhythmic peaks on the estimation of the linear background, the
    # following section removes these specified ranges
    freqKeep = np.array([True] *len(F))
    if excludePeak.size > 1: # allow for no peak removal
        for exFreqInd in range(excludePeak.shape[0]): # allow for multiple peaks
            # find empirical peak in specified range
            exFreq = excludePeak[exFreqInd]
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
    X = np.log10(fitInput['f_'])
    Y = np.mean(np.log10(fitInput['BG_']),axis = 1)
    X = sm.add_constant(X)
    rlm_model = sm.RLM(Y,X, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit()

    # perform the robust linear fit, only including putatively aperiodic components (i.e., peak exclusion)
    b = rlm_results.params

    
    pv = [b[1],b[0]]
    mp = 10**(b[0] + b[1]*np.log10(F))

    # compute eBOSC power (pt) and duration (dt) thresholds: 
    # power threshold is based on a chi-square distribution with df=2 and mean as estimated above
    from scipy.stats.distributions import chi2

    pt=chi2.ppf(threshold_pct,2)*mp/2 # chi2inv.m is part of the statistics toolbox of Matlab and Octave
    # duration threshold is the specified number of cycles, so it scales with frequency
    dt=threshold_duration*fsample/F;

    eBOSC = {}
    # save multiple time-invariant estimates that could be of interest:
    # overall wavelet power spectrum (NOT only background)
    eBOSC['bg_pow']        = np.mean(BG[:,total_sample:(-1-total_sample)],1)
    # log10-transformed wavelet power spectrum (NOT only background)
    eBOSC['bg_log10_pow'] = np.mean(np.log10(BG[:,total_sample:(-1-total_sample)]),1)
    # intercept and slope parameters of the robust linear 1/f fit (log-log)
    eBOSC['pv']            = pv
    # linear background power at each estimated frequency
    eBOSC['mp']          = mp
    # statistical power threshold
    eBOSC['pt']           = pt
    return eBOSC, pt, dt
