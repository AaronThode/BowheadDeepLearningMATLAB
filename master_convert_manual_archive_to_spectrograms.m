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
debug_plot=true;
debug.sec_to_load=60*60;

write_files=true;

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};
year_want={'10'};
Site={'5'};


sound_type='whale'; %whale, seal

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
            temp=datevec(tabs);
            temp(:,4:6)=0;
            tabs_start=datenum(temp);
            tabs_start_unique=unique(tabs_start);

            %%%Placeholder to read in clock drift information for GSI
            %%%file for this day...
            %if write_files

            if strcmpi(GSI_file_type,'gsi')
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

            elseif strcmpi(GSI_file_type,'wav')
                GSI_file_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0_WAV', ...
                    GSI_file_dir,year_want{Iyear},Site{Isite},year_want{Iyear}, ...
                    Site{Isite},year_want{Iyear},strr(Id));
                %fs=head.Fs*(1+head.tdrift/86400);
                if exist(GSI_file_want,'dir')~=7
                    GSI_file_want(end-5)='1';
                    if exist(GSI_file_want,'dir')~=7
                        disp('Data do not exist')
                        continue
                    end
                end
                GSI_names=dir([GSI_file_want '/S*WAV']);
            end
            %else
            %head.tdrift=0;
            %end


            for JJ=1:length(GSI_names)
                GSI_file_array{JJ}=GSI_names(JJ).name;
            end

            for Iday=1:length(tabs_start_unique)
                disp(datestr(tabs_start_unique(Iday)));

                Ifile_want=find(contains(GSI_file_array, datestr(tabs_start_unique(Iday),30)));


                %%%%%Import data%%%%%%%%
                fprintf('Reading %s\n',GSI_names(Ifile_want).name);
                %[x,headd]=(readgsi([GSI_file_want filesep GSI_names(Ifile_want).name],0,Inf));
                %x=int16(x(1,:)'-2^15);
                tic
                if strcmpi(GSI_file_type,'gsi')
                    [x,headd]=readgsi_omni_only([GSI_file_want filesep GSI_names(Ifile_want).name],0,debug.sec_to_load);
                else
                    [x,Fs]=audioread([GSI_file_want filesep GSI_names(Ifile_want).name],[1/1000 debug.sec_to_load]*1000,'native');

                end
                toc
                %x=int16(x-2^15);
                x=x-2^15;

                %[x,amp_scale]=calibrate_GSI_signal(x, 'DASARC');
                %amp_Scale = (2.5/65535)*(10^(149/20));

                %%%Process and save all manual detections
                sub_process_manual_detections;

                %%% Now generate false detections by a simple event
                %%% detector and check that they aren't whale calls.

            end %Iday
        end %Id
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear
