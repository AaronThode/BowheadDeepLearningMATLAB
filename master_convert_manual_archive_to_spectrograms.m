close all
clear
strr='ABCDEFG';
GSI_file_dir='/Volumes/Shared-2/Data/';
if exist(GSI_file_dir)==0
    error('GSI_file_dir not present')
end
debug_plot=false;


output_dir='WAV_snippets_Manual.dir';
year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};
sound_type='whale'; %whale, seal
write_files=true;
file_len_sec=5; %length of file clip in seconds.

%year_want={'08'};
Site={'5'};

for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        for I=1:length(strr)
            % DASAR_list{I}=['S314' strr(I) '0'];
            DASAR_list{I}=sprintf('S%s%s%s0',Site{Isite},year_want{Iyear},strr(I));
        end
        ctmin=0;
        ctmax=Inf;
        fname=sprintf('Shell_Manual_Results%s20%s%sAllSite%s_20%s_manual_archive.txt', ...
            filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});


        [ind,localized]=read_tsv_archive(fname,ctmin,ctmax,DASAR_list);

        if strcmpi(sound_type,'whale')
            Itype=find(localized.wctype<=7);  %bowhead whale calls only
        elseif strcmpi(sound_type,'seal')
            Itype=find(localized.wctype==8 | localized.wctype==9);  %seal and walrus only

        end

        if isempty(Itype)
            fprintf('No %s here\n',sound_type);
            continue
        end
        fieldnamess=fieldnames(ind);
        %ind.duration=ind.duration(Itype,:);
        for JJ=1:length(fieldnamess)
            ind.(fieldnamess{JJ})=ind.(fieldnamess{JJ})(Itype,:);
        end


        %%%Loop through dates and create a selection file for each DASAR and day

        for Id=1:size(ind.wgt,2)  %For each DASAR

            %keyboard
            fprintf('DASAR %s\n',DASAR_list{Id});
            tabs=datenum(1970,1,1,-8,0,ind.ctime(:,Id)); %-8 converts from UTC time (archive) to local time (GSI WAV)

            Ipass=find(~isnan(tabs));
            tabs=tabs(Ipass);
            temp=datevec(tabs);
            temp(:,4:6)=0;
            tabs_start=datenum(temp);
            tabs_start_unique=unique(tabs_start);

            %%%Placeholder to read in clock drift information for GSI
            %%%file for this day...
            if write_files
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
            else
                head.tdrift=0;
            end


            for JJ=1:length(GSI_names)
                GSI_file_array{JJ}=GSI_names(JJ).name;
            end

            for Iday=1:length(tabs_start_unique)
                 Ifile_want=find(contains(GSI_file_array, datestr(tabs_start_unique(Iday),30)));
               
                if Iday==1
                    fprintf('Reading %s\n',GSI_names(Ifile_want).name);
                    [x,headd]=(readgsi([GSI_file_want filesep GSI_names(Ifile_want).name],0,Inf));
                    x=int16(x(1,:)'-2^15);
                    %[x,amp_scale]=calibrate_GSI_signal(x, 'DASARC');
                    %amp_Scale = (2.5/65535)*(10^(149/20));
                    %x=x/amp_scale;  %%Convert to integer units
                end

                disp(datestr(tabs_start_unique(Iday)));
                Igood=find(tabs_start==tabs_start_unique(Iday));
                tsec=(tabs(Igood)-tabs_start_unique(Iday))*24*3600;
                tsec=tsec*(1+head.tdrift/86400);
                %output_name=datestr(tabs_start_unique(Iday),30);
                %output_name=sprintf('%sT%s.%s.Table.1.selections.txt',DASAR_list{Id},output_name,sound_type);

                if write_files
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

                    Igood_org=Ipass(Igood);  %Ensure that we skipp the NaN..
                    %Igood is associated with tabs.  
                    %Igood_org associated with original ind.* fields

                    for I=1:length(Igood)
                        tmid=tsec(I)+0.5*ind.duration(Ipass(Igood(I)),Id);
                        tsec_start=tmid-0.5*file_len_sec;
                        Ixx=round(head.Fs*[tsec_start+[0 file_len_sec]]);
                        y=x(Ixx(1):Ixx(2));
                        %tsec_end=tmid+0.5*file_len_sec;

                        if debug_plot
                            Nfft=256; spectrogram(double(y),Nfft,Nfft/2,Nfft,head.Fs,'yaxis')
                            clim([0 30])
                            title(sprintf('Filename: %s, middle time %6.2f seconds',GSI_names(Ifile_want).name,tmid))
                            pause(2)
                        end

                        output_name=GSI_names(Ifile_want).name(1:(end-4));
                        temp=datestr(tabs(Igood(I)),30);
                        output_name(17:end)=temp(10:end);
                        output_name=[output_dir filesep '20' year_want{Iyear} filesep 'Site' Site{Isite} filesep output_name '.wav'];
                        audiowrite(output_name,y,head.Fs,"BitsPerSample",16);
                        
                       
                    end
                    
                end %write_files
            end %Iday
        end %Id
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear
