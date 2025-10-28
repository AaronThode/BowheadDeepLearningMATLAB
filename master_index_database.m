%%%%master_index_database.m%%%%%
%   index{Iyear,Isite,Iday,Ifold,Idd}
%file_fraction.manual_all=zeros(length(year_want),length(Site),length(day_want),length(DASAR_ID),3);  %Last index is maximum number of 'D' folders expected.
close all
clear


database_folder='../Spectrogram_Image_Database.dir';
eval(sprintf('!mkdir %s/Trash.dir',database_folder));

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};

year_want={'14','10'};
Site={'5'};

DASAR_strings='ABCDEFG';
%day_want={'0815','0821','0829','0905','0913','0927'};
folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};
SNR_gram_dims=[121 104];

param.interval_remove.debug=false;

DASARs_counted=false;
max_D_dirs=0;
max_DASAR_strings='';
for Iyear=1:length(year_want)
    disp(year_want{Iyear});
    for Isite=1:length(Site)
        disp(Site{Isite});

        dir_string=sprintf('%s/20%s/Site%s/Day_*', ...
            database_folder,year_want{Iyear},Site{Isite});
        temp_dir_names=dir(dir_string);
        fprintf('Days in Year %s\n',year_want{Iyear});

        for Iday=1:length(temp_dir_names)
            day_want{Iday}=temp_dir_names(Iday).name(9:12);
            disp(day_want{Iday});
          
        end
        disp('Working though day list for this Site/year combo...')
        %%%Derives the unique day list
        for Iday=1:length(day_want)
            disp(day_want{Iday});
            for Ifold=1:length(folder_names)
                disp(folder_names{Ifold});
                dir_string=sprintf('%s/20%s/Site%s/Day_20%s%sT000000/%s', ...
                    database_folder,year_want{Iyear},Site{Isite},year_want{Iyear},day_want{Iday},folder_names{Ifold});
                DD_dirs=dir([dir_string '/D*dir']);
                max_D_dirs=max([max_D_dirs length(DD_dirs)]);
                for Idd=1:length(DD_dirs)
                    fnames=dir([dir_string '/' DD_dirs(Idd).name '/S*mat']);
                    Nfiles=length(fnames);

                    if Nfiles==0
                        disp([dir_string '/' DD_dirs(Idd).name ' has no files.']);
                        continue
                    end

                    str_length=length(fnames(1).name);
                    index{Iyear,Isite,Iday,Ifold,Idd}.fname=zeros(Nfiles,str_length);
                    index{Iyear,Isite,Iday,Ifold,Idd}.bearing=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold,Idd}.tabs=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold,Idd}.type=zeros(1,Nfiles);


                    for Ifile=1:Nfiles
                        temp=load([fnames(Ifile).folder filesep fnames(Ifile).name]);
                        if ~all(SNR_gram_dims==size(temp.SNR_gram))
                            disp([ fnames(Ifile).name ' SNR_gram not correct size:\n' size(temp.SNR_gram)]);
                            keyboard
                            eval(sprintf('!mv %s/%s %s/Trash.dir',fnames(Ifile).folder,fnames(Ifile).name,database_folder));
                            continue
                        end
                        index{Iyear,Isite,Iday,Ifold,Idd}.fname(Ifile,:)=fnames(Ifile).name;
                        index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Ifile)=temp.bearing;
                        index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Ifile)=temp.tabs_mid;
                    end
                    %%%Remove bad files from index
                    Igood=find(index{Iyear,Isite,Iday,Ifold,Idd}.tabs>0);
                    index{Iyear,Isite,Iday,Ifold,Idd}.fname=index{Iyear,Isite,Iday,Ifold,Idd}.fname(Igood,:);
                    index{Iyear,Isite,Iday,Ifold,Idd}.bearing=index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Igood);
                    index{Iyear,Isite,Iday,Ifold,Idd}.tabs=index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Igood);
                    index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun=zeros(1,length(Igood));
                   
                    %%%Identify DASARs present
                    DASAR_ID=char(unique(index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5)));
                    
                    if length(DASAR_ID)>length(max_DASAR_strings)
                        fprintf('new DASAR string is %s\n',DASAR_ID);
                        max_DASAR_strings=DASAR_ID;
                    end
                    if ~DASARs_counted
                        DASARs_counted=true;

                        file_fraction.manual_all=zeros(length(year_want),length(Site),length(day_want),length(DASAR_ID),3);  %Last index is maximum number of 'D' folders expected.
                        file_fraction.auto_all=file_fraction.manual_all;

                       % file_fraction.manual_noairgun=zeros(length(year_want),length(Site),length(day_want),length(DASAR_ID),4);  %Last index is maximum number of 'D' folders expected.
                        file_fraction.auto_noairgun=file_fraction.manual_all;

                    end

                     %%%Record number of files present here
                     %folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};
                     for Idasar=1:length(DASAR_ID)
                         Nfiles_DASAR=sum(double(DASAR_ID(Idasar))==index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5));
                         if Ifold==1  %auto
                             file_fraction.auto_all(Iyear,Isite,Iday,Idasar,Idd)=Nfiles_DASAR;
                         else  %manual calls
                             file_fraction.manual_all(Iyear,Isite,Iday,Idasar,Idd)=Nfiles_DASAR;
                         end
                     end

                     if Ifold==2
                         continue
                     end
                     %%%Optional search for airgun signals
                     disp('Starting airgun detection');tic
                     param.interval_remove.ICItol=[0.5 1.00];  %How close does a candidate have to be to predicted ICI time? (sec)
                     param.interval_remove.ICI_range=[7 10;
                         10 42];
                     param.interval_remove.Ndet=20;
                     param.interval_remove.Nmiss=12;
                     param.interval_remove.tol_feature=15;

                     param.interval_remove.ICI_std=0.4;  %How close do adjacent ICI estimates have to be (sec)
                     param.interval_remove.Nstd=7;      % How many ICIs must pass the 'ICI_std' test?

                     for Idasar=1:length(DASAR_ID)
                         Iltr=find(double(DASAR_ID(Idasar))==index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5));
                         param.interval_remove.titstr=fnames(Iltr(1)).name;
                         ICI=estimate_airgun_interval(index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Iltr),index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Iltr),param.interval_remove);
                         index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun(Iltr)=ICI;

                         Nfiles_DASAR=sum(ICI<1);
                         %  if Ifold==1  %auto
                         file_fraction.auto_noairgun(Iyear,Isite,Iday,Idasar,Idd)=Nfiles_DASAR;
                         %else  %manual calls
                         %    file_fraction.manual_noairgun(Iyear,Isite,Iday,Idasar,Idd)=Nfiles_DASAR;
                         %end
                     end %Idasar
                     disp('Finished airgun detection');toc
                    
                end %Idd

            end %Ifold
        end %Iday
    end %Isite
end %Iyear

%file_fraction.auto_all=file_fraction.auto_all/sum(file_fraction.auto_all(:));
%file_fraction.manual_all=file_fraction.manual_all/sum(file_fraction.manual_all(:));
%file_fraction.auto_noairgun=file_fraction.auto_noairgun/sum(file_fraction.auto_noairgun(:));
max_DASAR_strings=max_DASAR_strings.';
save([database_folder '/Database_index.mat'],'index','file_fraction','year_want','Site','day_want','folder_names','SNR_gram_dims','max_DASAR_strings');
