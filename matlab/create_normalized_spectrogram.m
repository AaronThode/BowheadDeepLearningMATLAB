function [BB_int,F,T]=create_normalized_spectrogram(y,Fs,spectrogram_len_sec,param)

[B,F,T] = spectrogram(y,param.Nfft,round(param.ovlap*param.Nfft),param.Nfft,Fs);
Ifreq=(F>=param.fmin & F<=param.fmax);
B=10*log10(abs(B(Ifreq,:)));
F=F(Ifreq);

dT=T(2)-T(1);
T_noise=(T(end)-spectrogram_len_sec)/2;

NN=round(T_noise/dT);
median_noise=median([B(:,1:NN) B(:,(length(T)-NN):length(T))],2);

Indexx=(NN+1):(length(T)-NN-1);
BB=(param.image_scale_factor)*(B(:,Indexx)-median_noise);
T=T(Indexx);
T=T-T(1);
if any(BB(:)>255)
    fprintf('SNR greater than %6.2f\n',255/param.image_scale_factor);
    keyboard
end
BB_int=uint8(BB);

end