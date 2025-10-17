%%%%%sub_plot_debug_manual_automated_comparison
%%%
%%%   Plot misses and matches between manual and automated results from
%%%   'master_convert_manual_archive_to_spectrogram'   


%%%%%%Examine missed manual detections
%param.spec.debug_plot=true;
%Itemp=Imiss;

if param.spec.debug_plot
    disp('Displaying missed manual detections')

    for I=1:length(Imiss)
        tmid=manual.tmid(Imiss(I));

        titstr{1}=sprintf('Missed manual detection: Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,tmid,I,length(Imiss));
        titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s',manual.SNR(Imiss(I)),datestr(manual.tabs(Imiss(I))));

        [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);

    end %I in Imiss
end %debug_plot

%%%Display automated detections that match%%%%%%%
param.spec.debug_plot_matches=false;
if param.spec.debug_plot_matches
    for Idet=1:length(Idet_match)

        II=Idet_match(Idet);
        tmid=0.5*(detect.tstart(II)+detect.tend(II));
        titstr{1}=sprintf('Matching Auto detect Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,tmid,Idet,length(Idet_match));
        titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s, score overlap: %6.4f', ...
            detect.dB_RMS(II),datestr(detect.tstart_abs(II)),Score{Ichunk}(II));
        [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);

    end %Idet
end %debug plot

