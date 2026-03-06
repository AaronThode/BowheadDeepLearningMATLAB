function  [outputs]=extract_features_from_SNRgram(dT,dF,SNR_gram);

image_scale_factor=5;
debug_plot=true;

FF=dF*(1:size(SNR_gram,1));
TT=dT*(1:size(SNR_gram,2));
tmp=max(SNR_gram,[],2);
[~,Imax]=max(tmp);
outputs.Fmax=FF(Imax);

tmp=max(SNR_gram,[],1);
[~,Imax]=max(tmp);
outputs.Tmax=TT(Imax);

SNR_gram=double(SNR_gram)/image_scale_factor;

w=10.^((SNR_gram-min(min(SNR_gram)))/10);
w_int=trapz(w);
mean_f=trapz(w.*FF')./w_int;
MWLB=median(2*sqrt(trapz(w.*((FF'-mean_f).^2))./w_int));

if debug_plot
    figure(101);
    imagesc(TT,FF,SNR_gram);%colorbar;
    ylim([0 500]);
    axis xy
    set(gca,'fontweight','bold','fontsize',14)
    title(sprintf('Peak Frequency: %6.2f Peak Time: %6.2f',outputs.Fmax,outputs.Tmax));
    colorbar
    keyboard
    close(101)
end
% pause