

%%%master_azigram_pulse_tracker.m%%%
clear
close all
!rm PulseTrackerLog.txt
diary PulseTrackerLog.txt

Nfft=256;
ovlap=0.90;

Scenario='Block2020';
param_chc='Aaron_standard'; %Aaron_standard,Aaron_permissive, SCUBA_track
azigram_param=get_azigram_param(param_chc);

save_result=false;  %If true, save *mat and *jpg files.
Fs=1000;
min_ts_fft = 2^nextpow2(Fs*azigram_param.max_time); % fixed fft size for TS

%%%Set these to true to see the morphological image processing and
%%%mas construction
azigram_param.debug.input=false;
azigram_param.debug.morph=false;
azigram_param.debug.detect=false;
azigram_param.debug.mask_construction=false;

%%%%%%How much data to read in at a time to RAM--solves clock drift
%%%%%%problem

azigram_param.time_chunk=30;

dBspread=[20 50];
azigram_param.debug.dBspread=dBspread;
azigram_param.sec_avg='0';


bearing_options={'med','avg'};

%%%%Get data directory, depending on computer this is running on.
data_dir=get_data_dir(Scenario);
[data_dir,mydir,fnames,base_ltr,pos_DASAR,xlimm,ylimm,tstart,tsample]=get_Scenario(Scenario,data_dir);
    
azigram_param.base_ltr=base_ltr;
Nstation=length(base_ltr);

Bfilt=[];
for I=1:length(fnames)  %Cycle through each day
    clear output
    try
        myfile{1}=fnames(I).name;
        fail=false;
        
        for Is=2:Nstation
            myfile{Is}=myfile{1};
            myfile{Is}(5)=base_ltr(Is);
        end
        
        for Ic=1:Nstation
            
           
            [x{Ic},~,head]=readGSIfile([data_dir mydir{Ic} '/' myfile{Ic}],tstart,tsample,1:3,'seconds','nocalibrate');
            x{Ic}=calibrate_GSI_signal(x{Ic}, 'DASARC')';
            azigram_param.brefa(Ic)=head.brefa;
            azigram_param.goodName{Ic}=[mydir{Ic} '/' myfile{Ic}];
            disp(myfile{Ic});
            %cd(current_dir)
        end
        
        
        %%%%Parameters for the automated processing
        azigram_param.tabs_start=head.tabs_start+datenum(0,0,0,0,0,tstart);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%azigram_pulse_detector call
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [output, TT,FF, azi, B, ItoE]=azigram_pulse_detector(x,Fs,Nfft,ovlap,azigram_param);
        fprintf('Finished processing, starting localization...\n');
        viewwindow=30;  %seconds
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%Localizations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        output.loc.avg=compute_pulse_positions(output.azi.avg,pos_DASAR,true);
        output.loc.med=compute_pulse_positions(output.azi.wm.med,pos_DASAR,true);
        output.loc.pos_DASAR=pos_DASAR;
        output.base_ltr=base_ltr;output.loc.xlimm=xlimm;output.loc.ylimm=ylimm;
        fprintf('Finished localization processing\n');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%Filter raw time series
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        maxfreq=azigram_param.frange(2);minfreq=azigram_param.frange(1);
        df=maxfreq-minfreq;
        if ~exist('Bfilt','var')||isempty(Bfilt)
            frange=[max([0 minfreq-0.1*df]) minfreq maxfreq min([maxfreq+0.1*df Fs/2])];
            [N,Fo,Ao,W] = firpmord(frange,[0 1 0],[0.05 0.01 0.05],Fs);
            Bfilt = firpm(N,Fo,Ao,W);
        end
        
        %%%%Bandpass filter data with no phase offset
        for Id=1:length(base_ltr)
            %xfilt{Id}=x{Id}-mean(x{Id},2);
            fprintf('Starting time series extraction of DASAR %s\n',base_ltr(Id));
            for KK=1:3
                x{Id}(KK,:)=filtfilt(Bfilt,1,x{Id}(KK,:)-mean(x{Id}(KK,:)));
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%Optional debug output of bounding boxes and other diagnostics
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
      
        if tsample<=azigram_param.time_chunk
            debug_localization_statistics;
            [output.features,azigram_param,Bfilt]=extract_wavelet_scalogram(x,output,azigram_param,base_ltr,Bfilt,Fs);
       
        end %tsample<azigram_chunk
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%  Extract time series and get more precise duration
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       [output.features,azigram_param,Bfilt]=extract_time_series_metrics_Aaron(x,output,azigram_param,base_ltr,Bfilt,Fs);
        
        %%%Computes ranges to each DASAR, plus the origin..
        
        for Idasar=1:Nstation
            output.loc.avg.ranges(Idasar,:)=sqrt((output.loc.avg.VM(1,:)-pos_DASAR(Idasar,1)).^2+(output.loc.avg.VM(2,:)-pos_DASAR(Idasar,2)).^2);
            output.loc.med.ranges(Idasar,:)=sqrt((output.loc.med.VM(1,:)-pos_DASAR(Idasar,1)).^2+(output.loc.med.VM(2,:)-pos_DASAR(Idasar,2)).^2);
        end
        
        title(sprintf('%s: %i hours',fnames(I).name,tsample/3600));
        drawnow;
        
        plot_bulk_2Dhistogram;
        
        if save_result
            Idot=findstr(myfile{1},'.gsi')-1;
            
            for Iname=1:length(base_ltr)
                savename_annotation{Iname}=sprintf('%s_%iminDur.mat',myfile{Iname}(1:Idot),tsample/60);
            end
            
            %%%%Convert detections into annotation files for manual review
            %success_flag=convert_automated_pulse_into_annotations(savename_annotation,output);
            savename=savename_annotation{end};
            
            It=min(strfind(savename,'T'))+1;
            tabs=datenum(savename(It:(It+14)),'yyyymmddTHHMMSS')+datenum(0,0,0,0,0,tstart);
            savename(It:(It+14))=datestr(tabs,30);
            savename = replace(savename,base_ltr(end),base_ltr);
            savename=replace(savename,'Dur.mat',sprintf('Dur_Sector%i_Thresh%i.mat',azigram_param.da,azigram_param.threshold));
            
            orient landscape
            print('-djpeg',[savename '.jpg']);
            
            save(savename,'output','azigram_param','pos_DASAR','base_ltr','tstart','tsample','bearing_options','xlimm','ylimm');
            fprintf('%s written!!! \n\n\n',savename);
        end
    catch ME
        
        fprintf('%s failed!!!!! \n\n\n\n',fnames(I).name);
        ME
        
    end
   
end  %fnames
diary off