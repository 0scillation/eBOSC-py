filename = 'demo_ebosc_py.fif';
hdr = ft_read_header(filename);
dat = ft_read_data(filename);

eegsignal = squeeze(dat(1,:,1));
F = 2.^[1:.125:7];
Fsample = 512;
wavenumber = 6;

%% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit
[B,T,F]=BOSC_tf(eegsignal,F,Fsample,wavenumber);

spectra = mean(log10(B),2);
plot(F(F<30),spectra(F<30),'--o')

%% Step 2: robust background power fit (see 2020 NeuroImage paper)
cfg.eBOSC.F  = F; 
cfg.eBOSC.wavenumber = 6;       
cfg.eBOSC.fsample = 512;           
cfg.eBOSC.pad.tfr_s  = 0;          
cfg.eBOSC.pad.detection_s   =0;    
cfg.eBOSC.pad.total_s   =0;      
cfg.eBOSC.pad.background_s   = 0;
cfg.eBOSC.pad.background_sample = 0;
cfg.eBOSC.pad.total_sample = 0;
cfg.eBOSC.threshold.excludePeak = [2 8]; 
cfg.eBOSC.threshold.duration    = 3;
cfg.eBOSC.threshold.percentile  = 0.95;
cfg.eBOSC.channel   = [];
cfg.eBOSC.trial    = 1;
cfg.eBOSC.trial_background = 1;

cfg.tmp.channel = 1;
eBOSC = [];

TFR.trial{1} = B;
[eBOSC, pt, dt] = eBOSC_getThresholds(cfg, TFR, eBOSC);


%% Step 3: detect rhythms and calculate Pepisode

% The next section applies both the power and the duration
% threshold to detect individual rhythmic segments in the continuous signals.
detected = zeros(size(TFR.trial{1}));
for f = 1:length(cfg.eBOSC.F)
    detected(f,:) = BOSC_detect(TFR.trial{1}(f,:),pt(f),dt(f),cfg.eBOSC.fsample);
end; clear f