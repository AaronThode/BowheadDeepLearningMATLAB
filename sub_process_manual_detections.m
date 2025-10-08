%%%%%Process manual detections%%%%%%%

%Igood_org=Ipass(Igood);  %Ensure that we skipp the NaN..
%Igood is associated with tabs.
%Igood_org associated with original manual.ind.* fields

for I=1:length(Igood)
    tmid=manual.tmid(I);
    tsec_start=tmid-0.5*file_len_sec;
    Ixx=round(head.Fs*(tsec_start+[0 file_len_sec]));

    Ixx(1)=max([1 (Ixx(1))]);
    Ixx(2)=min([length(x) Ixx(2)])-1;
    y=x(Ixx(1):Ixx(2));

    if length(y)~=head.Fs*file_len_sec
        disp('File length not right')
        continue
    end

    [SNR_gram,FF,TT]=create_normalized_spectrogram(y,head.Fs,spectrogram_len_sec,param.spec);

    if debug_plot
        subplot(2,1,1)
        spectrogram((y),param.spec.Nfft,param.spec.Nfft/2,param.spec.Nfft,head.Fs,'yaxis')
        clim([0 30]);colorbar
        title(sprintf('Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,tmid,I,length(Igood)))

        subplot(2,1,2)
        imagesc(TT,FF,SNR_gram);colorbar;axis xy
        title('Final SNR image')
        pause(2)
    end

    output_name=GSI_names(Ifile_want).name(1:(end-4));
    temp=datestr(tabs(Igood(I)),30);
    output_name(17:end)=temp(10:end);
    output_name=[output_dir filesep '20' year_want{Iyear} filesep 'Site' Site{Isite} filesep output_name '.mat'];
    save(output_name,'SNR_gram','FF','TT');
    %audiowrite(output_name,y,head.Fs,"BitsPerSample",16);
end %I in Igood