%%%%master_index_database.m%%%%%
close all
clear

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};

database_folder='../Spectrogram_Image_Database.dir';
eval(sprintf('!mkdir %s/Trash.dir',database_folder));
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

                    %%%Optional search for airgun signals
                    tsec=index{Iyear,Isite,Iday,Ifold}.tabs;
                    tsec=(tsec-tsec(1))*24*3600;

                    if debug.plot & strcmpi(folder_names{Ifold},'Event_sounds.dir')
                        figure(1);plot(index{Iyear,Isite,Iday,Ifold}.tabs,index{Iyear,Isite,Iday,Ifold}.bearing,'.')
                        xtickangle(90);
                        datetick('x',30)
                        keyboard
                    end

                end
            end
        end
    end
end