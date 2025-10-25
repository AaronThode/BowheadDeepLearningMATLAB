function [BB_int,F,T,median_raw]=create_normalized_spectrogram(y,Fs,spectrogram_len_sec,param)

param.debug_plot=true;
param.num_blobs=3;
param.blob_size=5;
%t1=tic;
median_raw=[];

if size(y,2)>1



    param.sec_avg=0;
    param.instrument= {'DASAR_DASAR-omnisensor'  'DASAR_DASAR-Xsensor'  'DASAR_DASAR-Ysensor'};
    [T,F,output_array,B,~,Ix,Iy]=compute_directional_metrics(y',{'Azimuth','ItoERatio'}, ...
        Fs,param.Nfft, param.ovlap, param);

    if param.debug_plot
        figure(1);set(gcf,'Position',[ 146     1   560   420])
        imagesc(T,F,B);title('Spectrogram');axis xy;colormap(jet);clim([0 40]); colorbar
        figure(2);set(gcf,'Position',[ 146   519   560   420])
        imagesc(T,F,output_array{1});title('Azigram');axis xy;colormap("hsv");colorbar

    end

    Ifreq=(F>=param.fmin & F<=param.fmax);
    B=B(Ifreq,:);

    dT=T(2)-T(1);
    T_noise=(T(end)-spectrogram_len_sec)/2;

    NN=round(T_noise/dT);
    median_noise=median([B(:,1:NN) B(:,(length(T)-NN):length(T))],2);

    Indexx=(NN+1):(length(T)-NN-1);
    SNR=(B(:,Indexx)-median_noise);


    azi_gram=output_array{1}(Ifreq,Indexx);
    NTV=output_array{2}(Ifreq,Indexx);
    T=T(Indexx);
    T=T-T(1);
    Wght=SNR.*NTV.^2;
    %Igood=find(Wght(:)>5)



    Igood = bwpropfilt(Wght>param.blob_size,'Area',param.num_blobs);
    angs=azi_gram(Igood);



    median_raw=circ_median(angs(:)*pi/180)*180/pi;
    median_raw(median_raw<0)=median_raw(median_raw<0)+360;


    azi_math=90 - angs;   %%%Angles now defined c-clockwise relative to x axis
    Ifix=(azi_math<-180);
    azi_math(Ifix)=azi_math(Ifix)+360;

    % compute median and std
    median_math=180*circ_median(azi_math*pi/180)/pi;  %Original existing line
    iqr_out=180*circ_std(azi_math*pi/180)/pi;

    %Convert back into compass convention.
    median_compass=90-median_math;
    median_compass(median_compass<0)=median_compass(median_compass<0)+360;

    BB_int=param.image_scale_factor.*uint8(SNR);



    if param.debug_plot
        figure(3);set(gcf,'Position',[ 663          41        1122         893])

        subplot(3,2,1)
        imagesc(T,F,SNR);title('SNR');axis xy;colormap(jet);colorbar;clim([5 30])
        subplot(3,2,2)
        imagesc(T,F,NTV);title('NTV');axis xy;clim([0 1]);colorbar
        subplot(3,2,3)
        imagesc(T,F,Wght);title('NTV*SNR^2');axis xy;clim([5 30]);colorbar
        subplot(3,2,4)
        imagesc(T,F,Igood);title(sprintf('top %i blobs',param.num_blobs));axis xy;colorbar
        subplot(3,2,5)
        imagesc(T,F,Igood.*azi_gram);title('filtered for top blobs');axis xy;colorbar;clim([0 360]);colorbar

        stats=regionprops(Wght>param.blob_size,'Area');Area=[stats.Area];
        subplot(6,2,10);plot(Area,'x');grid on;ylabel('Area(pixels)');%xlabel('Blob index')
        % figure;
        subplot(6,2,12);histogram(azi_gram(Igood));
        title(sprintf('Raw: %6.2f, Compass %6.2f',median_raw,median_compass),'Interpreter','none')
        %keyboard
        pause;
        close all
    end



    %   Output:
    %    TT,FF, output_array{Nmetric}:  TT vector of times and FF vector of Hz for
    %                 output_array grid, one for each value in metric_type.
    %                 If 'polarization' is a metric, output_array{}[freq times metric] will have
    %                 three dimensions, with third dimension being Stokes metric.

    % [thet0,kappa,sd]=extract_bearings(x{II}',param.energy.bufferTime,param.Nfft,param.Fs,fmin(II),fmax(II),run_options.bearing_alg,2);
    % thet(Iref(II))=bnorm(thet0+head.brefa);




else
    [B,F,T] = spectrogram(y(:,1),param.Nfft,round(param.ovlap*param.Nfft),param.Nfft,Fs);
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

%toc(t1)

end