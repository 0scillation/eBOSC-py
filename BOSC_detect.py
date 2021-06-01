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
def BOSC_detect(b,powthresh,durthresh,Fsample):
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