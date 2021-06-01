# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:33:19 2021

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

def BOSC_tf(eegsignal,F,Fsample,wavenumber):
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