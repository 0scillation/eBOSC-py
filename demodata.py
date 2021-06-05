# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:54:08 2021

@author: Lemon
"""

#cross validation with matlab BOSC
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
cd "C:/Users/Lemon/Documents/Github/eBOSC-py"
import BOSC 

data_folder = Path("C:/Users/Lemon/Documents/Github/eBOSC-py")
demodata = mne.read_epochs(data_folder / "demo_ebosc_py.fif")
demodata = demodata.get_data()
eegsignal = demodata[0,0,:]

F = 2**np.linspace(1,7,num=49)
Fsample = 512
wavenumber = 6
#%% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit
B=BOSC.tf(eegsignal,F,Fsample,wavenumber)

spectra = np.mean(np.log10(B),1)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(F,spectra,label='Demo')

#%% Step 2: robust background power fit (see 2020 NeuroImage paper)
cfg = {}
cfg['trial_background'] = 'all'
cfg['background_sample'] = 0
cfg['total_sample'] = 0
cfg['excludePeak'] = np.asarray(([2,8]))
cfg['wavenumber'] = 6
cfg['threshold_pct'] = 0.95
cfg['threshold_duration'] = 3
cfg['fsample'] = 512
cfg['F'] = F

eBOSC = [];

[eBOSC, pt, dt] = BOSC.getThresholds(cfg, B, eBOSC);
