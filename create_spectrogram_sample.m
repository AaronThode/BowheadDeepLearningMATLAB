function [SNR_gram,FF,TT]=create_snippet(x,Fs,tmid,file_len_sec,spectrogram_len_sec,param,titstr)
%[SNR_gram,FF,TT]=create_snippet(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr)

SNR_gram=[];TT=[];FF=[];
tsec_start=tmid-0.5*file_len_sec;
Ixx=round(Fs*(tsec_start+[0 file_len_sec]));

Ixx(1)=max([1 (Ixx(1))]);
Ixx(2)=min([length(x) Ixx(2)])-1;
y=x(Ixx(1):Ixx(2));

if length(y)~=Fs*file_len_sec
    disp('File length not right')
    return
end

[SNR_gram,FF,TT]=create_normalized_spectrogram(y,Fs,spectrogram_len_sec,param);

if param.debug_plot
    figure(1)
    subplot(2,1,1)
    spectrogram((y),param.Nfft,param.Nfft/2,param.Nfft,Fs,'yaxis')
    clim([0 30]);colorbar
   %title(sprintf('Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,tmid,I,length(Itemp)))
    title(titstr{1})
    subplot(2,1,2)
    imagesc(TT,FF,SNR_gram);colorbar;axis xy
    title(titstr{2})
    %title(sprintf('Final SNR image, SNR: %6.2f, abs start: %s',manual.SNR(I),datestr(manual.tabs(I))));
    pause
end


end