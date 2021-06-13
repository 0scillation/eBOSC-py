filename = 'test.fif';
hdr = ft_read_header(filename);
dat = ft_read_data(filename);
dat = dat(1,:,:);
dat = squeeze(dat);
%%
eegsignal = squeeze(dat(1,:,1));
F = 2.^[1:.125:6];
Fsample = 512;
wavenumber = 6;

%% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit
B = nan(41,2561,24);
for itrial = 1:24
    eegsignal = dat(:,itrial);
    [B(:,:,itrial),T,F]=BOSC_tf(eegsignal,F,Fsample,wavenumber);
end
% spectra = mean(log10(B),2);
% plot(F(F<30),spectra(F<30),'--o')
%% Step 2: robust background power fit (see 2020 NeuroImage paper)
percentilethresh = 0.95;
numcyclesthresh = 3;
[pv,meanpower]=BOSC_bgfit(F,mean(B,3));
[powthresh,durthresh]=BOSC_thresholds(Fsample,percentilethresh,numcyclesthresh,F,meanpower);

%% Step 3: detect rhythms and calculate Pepisode

% The next section applies both the power and the duration
% threshold to detect individual rhythmic segments in the continuous signals.
detected = zeros(size(B));
for itrial = 1:24
    for f = 1:length(F)
        detected(f,:,itrial) = BOSC_detect(B(f,:,itrial),powthresh(f),durthresh(f),Fsample);
    end
end
clear f
%%
pepisode = mean(mean(detected,2),3);
figure(1)
plot(F,pepisode)

%%
figure(3)
for itrial = 1:24
    plot(dat(:,itrial)*1000000)
    pause
end
%%
figure(2)
y = log10(mean(mean(B,2),3));
plot(F,y)

%% with original BOSC
% Supplementary Figure: plot estimated background + power threshold
figure; hold on;
plot(log10(F), log10(meanpower), 'k--','LineWidth', 1.5); 
plot(log10(F), log10(powthresh), 'k-', 'LineWidth', 1.5)
plot(log10(F), log10(mean(mean(B,2),3)), 'r-', 'LineWidth', 2)
xlabel('Frequency (log10 Hz)'); ylabel('Power (log 10 a.u.)');
legend({'Aperiodic fit', 'Statistical power threshold', 'Avg. spectrum'}, ...
    'orientation', 'vertical', 'location', 'SouthWest'); legend('boxoff');
set(findall(gcf,'-property','FontSize'),'FontSize',20)
