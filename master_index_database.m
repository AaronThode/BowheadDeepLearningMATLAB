%%%%master_index_database.m%%%%%
close all
clear


database_folder='../Spectrogram_Image_Database.dir';
eval(sprintf('!mkdir %s/Trash.dir',database_folder));

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};

year_want={'10'};
Site={'5'};
DASAR_strings='ABCDEFG';
day_want={'0815','0821','0829','0905','0913','0927'};
folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};
SNR_gram_dims=[128 104];

debug.plot=true;

for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        for Iday=1:length(day_want)
            for Ifold=1:length(folder_names)
                dir_string=sprintf('%s/20%s/Site%s/Day_20%s%sT000000/%s', ...
                    database_folder,year_want{Iyear},Site{Isite},year_want{Iyear},day_want{Iday},folder_names{Ifold});
                DD_dirs=dir([dir_string '/D*dir']);
                for Idd=1:length(DD_dirs)
                    fnames=dir([dir_string '/' DD_dirs(Idd).name '/S*mat']);
                    Nfiles=length(fnames);

                    if Nfiles==0
                        disp([dir_string '/' DD_dirs(Idd).name ' has no files.']);
                        continue
                    end

                    str_length=length(fnames(1).name);
                    index{Iyear,Isite,Iday,Ifold}.fname=zeros(Nfiles,str_length);
                    index{Iyear,Isite,Iday,Ifold}.bearing=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold}.tabs=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold}.type=zeros(1,Nfiles);


                    for Ifile=1:Nfiles
                        temp=load([fnames(Ifile).folder filesep fnames(Ifile).name]);
                        if ~all(SNR_gram_dims==size(temp.SNR_gram))
                            disp([fnames(Ifile).folder filesep fnames(Ifile).name 'SNR_gram not correct size:' size(temp.SNR_gram)]);
                            eval(sprintf('!mv %s %s/Trash.dir',fnames(Ifile).name,database_folder));
                            pause
                            continue
                        end
                        index{Iyear,Isite,Iday,Ifold}.fname(Ifile,:)=fnames(Ifile).name;
                        index{Iyear,Isite,Iday,Ifold}.bearing(Ifile)=temp.bearing;
                        index{Iyear,Isite,Iday,Ifold}.tabs(Ifile)=temp.tabs_mid;
                    end
                    %%%Remove bad files from index
                    Igood=find(index{Iyear,Isite,Iday,Ifold}.tabs>0);
                    index{Iyear,Isite,Iday,Ifold}.fname=index{Iyear,Isite,Iday,Ifold}.fname(Igood,:);
                    index{Iyear,Isite,Iday,Ifold}.bearing=index{Iyear,Isite,Iday,Ifold}.bearing(Igood);
                    index{Iyear,Isite,Iday,Ifold}.tabs=index{Iyear,Isite,Iday,Ifold}.tabs(Igood);
                    index{Iyear,Isite,Iday,Ifold}.is_airgun=zeros(1,length(Igood));

                    %%%Optional search for airgun signals
                    tabs=index{Iyear,Isite,Iday,Ifold}.tabs;
                    tsec=(tabs-tabs(1))*24*3600;
                    bearing=index{Iyear,Isite,Iday,Ifold}.bearing;
                    %%%Identify DASARs present
                    DASAR_ID=char(unique(index{Iyear,Isite,Iday,Ifold}.fname(:,5)));
                    for Id=1:length(DASAR_ID)
                        Iltr=find(double(DASAR_ID(Id))==index{Iyear,Isite,Iday,Ifold}.fname(:,5));

                        if debug.plot & strcmpi(folder_names{Ifold},'Event_sounds.dir')
                            figure(1);
                            subplot(length(DASAR_ID),1,Id)
                            plot(index{Iyear,Isite,Iday,Ifold}.tabs(Iltr),index{Iyear,Isite,Iday,Ifold}.bearing(Iltr),'.')
                            xtickangle(90);grid on
                            datetick('x',30)
                        end

                        %%Compute ICI (regular intervals) from raw data, using selected features
                        %%for assistance.
                       
                        param.interval_remove.ICItol=[0.5 1.00];  %How close does a candidate have to be to predicted ICI time? (sec)
                        param.interval_remove.ICI_range=[7 10;
                            10 42];
                        param.interval_remove.Ndet=20;
                        param.interval_remove.Nmiss=12;
                        param.interval_remove.tol_feature=15;

                        param.interval_remove.ICI_std=0.4;  %How close do adjacent ICI estimates have to be (sec)
                        param.interval_remove.Nstd=7;      % How many ICIs must pass the 'ICI_std' test?

                        try
                            ICI=compute_ici_bothways_feature(tabs(Iltr),param.interval_remove.ICI_range, ...
                                bearing(Iltr), {'bearing'},...
                                param.interval_remove.Ndet,param.interval_remove.Nmiss,...
                                param.interval_remove.tol_feature,param.interval_remove.ICItol);


                            %function [ICI]=compute_ici_bothways_feature(tabs,ici_range,feature_array,feature_names, ...
                            %       num_clicks,num_misses,tol_feature,tol_time,Idebug),
                            %
                            %%% Given a vector of datenumbers, search for regular occurrances and
                            %%% estimate inter-click interval (ICI) and multipath
                            %    tabs: [1 x Ntimes] vector of datenumbers
                            %       to enable the first detected sound to be assigned an ICI
                            %    ici_range=[0.1 2; 5 10]; %Range of ICIs to test for in seconds.  If
                            %       multiple rows, test each range
                            %    feature_array: [Nfeature x Ntimes] array of features
                            %    feature_names: cell array of feature names, corresponds with rows of
                            %       feature array.
                            %    num_clicks=3;  %number of adjacent detections to examine for ICI
                            %    num_misses=1;  %How many "gaps" can exist in the ICI trace
                            %    tol_feature: [Nfeature x 1] array of tolerances (absolute) for feature
                            %       matching
                            %    tol_time=[0.01; 0.5];  %Tolerance for accepting a detection if close to ICI,(sec)
                            %       The number of rows much match the number of rows in ici_range.
                            %    Idebug:
                            %       .fname: raw GSI file from which data extracted.
                            %       .names:  cell array (Nfeature cells) with descriptive string of
                            %           features in feature_array.
                            %       .index:  index of element of tabs to begin plotting debug output.
                            %
                        catch
                            disp('Check your MATLAB path: ICI_detectors may be missing');
                            keyboard
                        end

                        if debug.plot & strcmpi(folder_names{Ifold},'Event_sounds.dir')
                            figure(2+10*Iday+Id)

                            subplot(2,2,1)
                            plot(tabs(Iltr),ICI,'x');
                            xtickangle(90);grid on
                            datetick('x',30)
                            title(fnames(Iltr(1)).name,'interp','none')

                            subplot(2,2,3)
                            plot(bearing(Iltr),ICI,'x');grid on;xlabel('Azimuth (deg)');ylabel('ICI (sec)');

                        end
                        %%Added April 5, 2010:  Sometimes walrus sequences and heavy bowhead whale
                        %%sequences can have an ICI, but the ICI is inconsistent between calls.
                        %  Thus here we march through each ICI detection and check whether detections
                        %  nearby share the same ICI.

                        Iguns=find(ICI>0);

                        ICI_score=ones(size(ICI));
                        for I=1:length(Iguns)
                            current_time=tsec(Iltr(Iguns(I)));
                            current_ICI=ICI(Iguns(I));
                            Itest=find(abs(current_time-tsec(Iltr(Iguns)))<=0.5*param.interval_remove.Ndet*current_ICI);
                            Ipass=0;
                            for J=1:2  %Harmonic loop:  checks for possibility that a 10 s ICI may have been assigned a 20 s ICI.
                                Ipass=Ipass+length(find(abs(ICI(Iguns(Itest))/J-current_ICI)<=param.interval_remove.ICI_std));
                            end

                            %%Are there enough matching ICIs close to the value of the current ICI?
                            if (Ipass-1)<param.interval_remove.Nstd
                                ICI_score(Iguns(I))=0;
                            else
                                %disp('good');
                            end

                        end

                        ICI=ICI.*ICI_score;

                        if debug.plot & strcmpi(folder_names{Ifold},'Event_sounds.dir')
                            %figure(2)
                            subplot(2,2,2)
                            plot(tabs(Iltr(ICI>0)),ICI(ICI>0),'x');
                            xtickangle(90);grid on
                            datetick('x',30)
                            title(fnames(Iltr(1)).name,'interp','none')

                            subplot(2,2,4)
                            plot(bearing(Iltr),ICI,'x');grid on;xlabel('Azimuth (deg)');ylabel('ICI (sec)');
                            pause
                        end

                        index{Iyear,Isite,Iday,Ifold}.is_airgun(Iltr)=ICI;
                    end %Id

                end

            end
        end
    end
end

save([database_folder '/Database_index.mat'],'index','year_want','Site','day_want','folder_names','SNR_gram_dims');
