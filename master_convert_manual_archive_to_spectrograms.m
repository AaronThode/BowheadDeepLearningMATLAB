%%%%%master_convert_manual_archive_to_spectrograms.m%%%
%
%
% Key points to remember when making spectrograms:
%       (1) They must be equalized;
%       (2) If not equalized they must be calibrated by the DASAR
%       calibation curve.
%       (3) They must be saved in uint16 to save as much space as possible.

close all
clear
strr='ABCDEFG';
GSI_file_dir='/Volumes/Shared/Data/';
%GSI_file_dir='/Volumes/Bowhead4/';
GSI_file_type='GSI';
Manual_record_files_dir='../Shell_Manual_Results';
output_dir='../Manual_sample_database.dir';

if exist(GSI_file_dir)==0
    error('GSI_file_dir not present')
end
debug_plot=false;
debug.sec_to_load=1*60*60;
debug.Iday_start=2;

%write_files=true;

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};
year_want={'10'};
Site={'5'};


sound_type='whale'; %whale, seal

%%%% Seconds of data to convert to spectrogram to conserve RAM memory.
%%%%   Duration should be short enough that background noise not expected
%%%%   to change...

chunk_sample=60*60;  %seconds
file_len_sec=10; %length of final file clip (includes noise estimate)
spectrogram_len_sec=5; %length of final spectrogram clip. (data used for noise removed)

%%%Parameters for event detection
param.event.dB_threshold = 20; % threshold above mean for detection
param.event.image_scale_factor = 5;  % factor to multiply SNR by for saving as unit8 image
param.event.fmin = 10;
param.event.fmax = 475;

param.spec.Nfft=256;
param.spec.ovlap=0.75;
param.spec.image_scale_factor = param.event.image_scale_factor;
param.spec.fmin = param.event.fmin;
param.spec.fmax = param.event.fmax;

nu=1.7;


for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        for I=1:length(strr)
            % DASAR_list{I}=['S314' strr(I) '0'];
            DASAR_list{I}=sprintf('S%s%s%s0',Site{Isite},year_want{Iyear},strr(I));
        end
        ctmin=0;
        ctmax=Inf;
        fname=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.txt', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});

        fname_mat=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.mat', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});

        if ~exist(fname_mat,'file')
            [ind,localized]=read_tsv_archive(fname,ctmin,ctmax,DASAR_list);
            save(fname_mat,'ind','localized');
            clear ind localized
        end
        manual=load(fname_mat);

        if strcmpi(sound_type,'whale')
            Itype=find(manual.localized.wctype<=7);  %bowhead whale calls only
        elseif strcmpi(sound_type,'seal')
            Itype=find(manual.localized.wctype==8 | manual.localized.wctype==9);  %seal and walrus only

        end

        if isempty(Itype)
            fprintf('No %s here\n',sound_type);
            continue
        end
        fieldnamess=fieldnames(manual.ind);
        %manual.ind.duration=manual.ind.duration(Itype,:);
        for JJ=1:length(fieldnamess)
            manual.ind.(fieldnamess{JJ})=manual.ind.(fieldnamess{JJ})(Itype,:);
        end


        %%%Loop through dates and create a selection file for each DASAR and day

        for Id=1:size(manual.ind.wgt,2)  %For each DASAR

            %keyboard
            fprintf('DASAR %s\n',DASAR_list{Id});
            tabs=datenum(1970,1,1,-8,0,manual.ind.ctime(:,Id)); %-8 converts from UTC time (archive) to local time (GSI WAV)

            Ipass=find(~isnan(tabs));
            tabs=tabs(Ipass);
            SIG_all=manual.ind.sigdb(Ipass,Id);
            SNR_all=manual.ind.stndb(Ipass,Id);
               
            temp=datevec(tabs);
            temp(:,4:6)=0;
            tabs_start=datenum(temp);
            tabs_start_unique=unique(tabs_start);
 
            %%%Placeholder to read in clock drift information for GSI
            %%%file for this day...
           
            GSI_file_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0', ...
                GSI_file_dir,year_want{Iyear},Site{Isite},year_want{Iyear}, ...
                Site{Isite},year_want{Iyear},strr(Id));
            %fs=head.Fs*(1+head.tdrift/86400);
            if exist(GSI_file_want,'dir')~=7
                GSI_file_want(end)='1';
                if exist(GSI_file_want,'dir')~=7
                    disp('Data do not exist')
                    continue
                end
            end
            GSI_names=dir([GSI_file_want '/*gsi']);
            head=readgsif_header([GSI_file_want filesep GSI_names(1).name]);


            for JJ=1:length(GSI_names)
                GSI_file_array{JJ}=GSI_names(JJ).name;
            end

            for Iday=debug.Iday_start:length(tabs_start_unique)
                disp(datestr(tabs_start_unique(Iday)));

                Igood=find(tabs_start==tabs_start_unique(Iday));
                manual.tabs=tabs(Igood);
                manual.tsec=(tabs(Igood)-tabs_start_unique(Iday))*24*3600;
                manual.tsec=manual.tsec*(1+head.tdrift/86400);
                manual.duration=manual.ind.duration(Ipass(Igood),Id);
                manual.tmid=manual.tsec+0.5*manual.duration;
                manual.tend=manual.tsec+manual.duration;
                manual.SNR=SNR_all(Igood);
                manual.sig=SIG_all(Igood);

                mydir=pwd;
                cd(output_dir)
                if Iday==1
                    eval(sprintf('!mkdir 20%s', year_want{Iyear}));
                end
                cd(sprintf('20%s',year_want{Iyear}));
                if Iday==1
                    eval(sprintf('!mkdir Site%s',Site{Isite}));
                end
                cd(mydir)

                Ifile_want=find(contains(GSI_file_array, datestr(tabs_start_unique(Iday),30)));

                %%%%%Import data%%%%%%%%
                fprintf('Reading %s\n',GSI_names(Ifile_want).name);
                %[x,headd]=(readgsi([GSI_file_want filesep GSI_names(Ifile_want).name],0,Inf));
                %x=int16(x(1,:)'-2^15);
                tic
                if strcmpi(GSI_file_type,'gsi')
                    [x,~,head]=readgsi_omni_only([GSI_file_want filesep GSI_names(Ifile_want).name],0,debug.sec_to_load);
                else
                    [x,Fs]=audioread([GSI_file_want filesep GSI_names(Ifile_want).name],[1/1000 debug.sec_to_load]*1000,'native');

                end
                toc
                %x=int16(x-2^15);
                x=x-2^15;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%Process and save all manual detections
                
                disp('Starting manual spectrograms')
                Itemp=Igood;
                sub_process_manual_detections;
                clear Itemp
                disp('Finished manual spectrograms')
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%Energy Detector.m%%%%%%%
                %%% Now generate false detections by a simple event
                %%% detector and check that they aren't whale calls.
                %
                %First, to save memory we will load data into chunks.
                %Chunks must be short enough that the background spectrum
                %does not evolve substantially over the interval
                %chunk_sample=60;  

                 %Set up energy detector.  Example for bowhead whale analysis that monitors between 25 and 350 Hz,
                %  using a set of detectors with 37 Hz bandwidth.
                K=1;
                param.energy.Nfft=256;
                param.energy.Fs =head.Fs;
                param.energy.ovlap = 0.75;
                param.energy.flo_det=25;
                param.energy.fhi_det=450;
                param.energy.burn_in_time=0.5;  %Time in minutes
                param.energy.eq_time=10;   param.energy_desc{K}='Equalization time (s): should be roughly twice the duration of signal of interest';K=K+1;
                param.energy.bandwidth=37;     param.energy_desc{K}='Bandwidth of sub-detector in kHz';K=K+1;
                param.energy.threshold=5;  param.energy_desc{K}='Threshold in dB to accept a detection';K=K+1;
                param.energy.TolTime=1e-4;  param.energy_desc{K}='Minimum time in seconds that must elapse for two detections to be listed as separate';K=K+1;
                param.energy.MinTime=0;     param.energy_desc{K}='Minimum time in seconds a required for a detection to be logged';K=K+1;
                param.energy.MaxTime=5;     param.energy_desc{K}= 'Maximum time in seconds a detection is permitted to have';K=K+1;
                param.energy.debug=0;       param.energy_desc{K}= '0: do not write out debug information. 1:  SEL output.  2:  equalized background noise. 3: SNR.';K=K+1;


                Nchunks=floor(length(x)/(chunk_sample*head.Fs));
                for Ichunk=1:Nchunks
                     Iss=1+(Ichunk-1)*chunk_sample*head.Fs;
                     x_chunk=x(Iss:(Iss-1+chunk_sample*head.Fs));
                     [detect,debugg]=MultipleBandEnergyDetector(x_chunk,head.tabs_start+datenum(0,0,0,0,0,(Ichunk-1)*chunk_sample),param.energy);
                     detect.tstart=detect.tstart+(Ichunk-1)*chunk_sample;
                     detect.tend=detect.tend+(Ichunk-1)*chunk_sample;
                     %detect.tmid_abs=0.5*(detect.tstart_abs+detect.tend_abs);

                      %%%Determine whether any overlap exists between
                         %%%manual detections and these detections.
                         %%%make comparisons in terms of absolute times.
                        
                     param.compare.ovlap=0.5;
                     [Score{Ichunk},Manual_index]=evaluate_overlap_between_manual_automated(manual.tsec,manual.tend,detect.tstart,detect.tend,param.compare.ovlap);
                %end

                     Idet_match=find(Score{Ichunk}>0);
                     Manual_index_match=Manual_index(Score{Ichunk}>0);
                     Imiss=setdiff(1:max(Manual_index_match),Manual_index_match);
                     fprintf('%i out of %i (%6.2f percent) manual detections missing from automated detections\n',length(Imiss),max(Manual_index_match),100*length(Imiss)/max(Manual_index_match))
                     

                     %%%%%%Examine missed manual detections
                     debug_plot=true;
                     Itemp=Imiss;
                     disp('Displaying missed manual detections')
                     sub_process_manual_detections;

                     for Idet=1:length(Idet_match)

                         II=Idet_match(Idet);
                         
                         Imid=round(head.Fs*0.5*(detect.tstart(II)+detect.tend(II)));
                         Ixx=Imid+0.5*file_len_sec*head.Fs*[-1 1];
                         
                         Ixx(1)=max([1 (Ixx(1))]);
                         Ixx(2)=min([length(x) Ixx(2)])-1;
                         y=x(Ixx(1):Ixx(2));
                         if length(y)~=head.Fs*file_len_sec
                             disp('File length not right')
                             continue
                         end

                         [SNR_gram,FF,TT]=create_normalized_spectrogram(y,head.Fs,spectrogram_len_sec,param.spec);
                            
                         if debug_plot
                             figure(2);
                             subplot(2,1,1)
                             spectrogram((y),param.spec.Nfft,param.spec.Nfft/2,param.spec.Nfft,head.Fs,'yaxis')
                             clim([0 30]);colorbar
                             title(sprintf('Auto detect Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,Imid/head.Fs,Idet,length(Idet_match)))

                             subplot(2,1,2)
                             imagesc(TT,FF,SNR_gram);colorbar;axis xy
                             title('Final SNR image')
                             title(sprintf('Final SNR image, SNR: %6.2f, abs start: %s, score overlap: %6.4f', ...
                                 detect.dB_RMS(II),datestr(detect.tstart_abs(II)),Score{Ichunk}(II)));

                             pause
                         end

                     end %Idet

                end %Ichunk
               
             

            end %Iday
        end %Id
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear
