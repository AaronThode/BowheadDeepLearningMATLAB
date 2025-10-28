%%%%master_index_database.m%%%%%
close all
clear


database_folder='../Spectrogram_Image_Database.dir';
data=load([database_folder '/Database_index.mat']);
eval(sprintf('!mkdir %s/Unsupervised_database.dir',database_folder));
eval(sprintf('!rm %s/Unsupervised_database.dir/*.mat',database_folder));

Nsamples=5000;
manual_fraction=0.50;
include_airguns=true;

year_want={'14','10'};
Site={'5'};
DASAR_strings={'A','C','G'};
%day_want={'0905','0913','0927'};
folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};

%   index{Iyear,Isite,Iday,Ifold,Idd}
%file_fraction.manual_all=zeros(length(year_want),length(Site),length(day_want),length(DASAR_ID),3);  %Last index is maximum number of 'D' folders expected.

for I=1:length(data.max_DASAR_strings)
    data.DASAR_strings{I}=data.max_DASAR_strings(I);
end
folder_fractions=[1-manual_fraction manual_fraction];
%%Trim down databases to match desired ranges above
Isite=contains(data.Site,Site);
Iyear=contains(data.year_want,year_want);
%Iday=contains(data.day_want,day_want);
Idasar=contains(data.DASAR_strings,DASAR_strings);
data_sub.index=data.index(Iyear,Isite,:,:,:);
data_sub.file_fraction.manual=data.file_fraction.manual_all(Iyear,Isite,:,Idasar,:);
if include_airguns
    data_sub.file_fraction.auto=data.file_fraction.auto_all(Iyear,Isite,:,Idasar,:);
else
    data_sub.file_fraction.auto=data.file_fraction.auto_noairgun(Iyear,Isite,:,Idasar,:);
end
data_sub.file_fraction.auto=data_sub.file_fraction.auto/sum(data_sub.file_fraction.auto(:));
data_sub.file_fraction.manual=data_sub.file_fraction.manual/sum(data_sub.file_fraction.manual(:));

for Iyear=1:length(year_want)
    disp(year_want{Iyear});
    for Isite=1:length(Site)
        disp(Site{Isite});

        dir_string=sprintf('%s/20%s/Site%s/Day_*', ...
            database_folder,year_want{Iyear},Site{Isite});
        temp_dir_names=dir(dir_string);
        for Iday=1:length(temp_dir_names)
            day_want{Iday}=temp_dir_names(Iday).name(9:12);
        end
        %%%Derives the unique day list
        for Iday=1:length(day_want)
            disp(day_want{Iday});
            for Ifold=1:length(folder_names)
                disp(folder_names{Ifold});
                dir_string=sprintf('%s/20%s/Site%s/Day_20%s%sT000000/%s', ...
                    database_folder,year_want{Iyear},Site{Isite},year_want{Iyear},day_want{Iday},folder_names{Ifold});
                if Ifold==1
                    file_fraction=data_sub.file_fraction.auto;
                else
                    file_fraction=data_sub.file_fraction.manual;
                end
                Nsamples_want=floor(folder_fractions(Ifold)*Nsamples*squeeze(file_fraction(Iyear,Isite,Iday,:,:)));

                Ndd=size(Nsamples_want,2);
                for Idd=1:Ndd
                    if sum(Nsamples_want(:,Idd))==0
                        continue
                    end

                    fnames=data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.fname;
                    %%%Remove airgun signals if desired
                    if ~include_airguns
                        fnames=fnames(data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun<1,:);
                    end
                    for Idasar=1:length(DASAR_strings)
                        disp(DASAR_strings{Idasar});
                        Igood=find(fnames(:,5)==DASAR_strings{Idasar});
                        Nfiles_local=length(Igood);
                        if Nfiles_local==0
                            continue
                        end
                        %Two situations can occur.  First, the number
                        %of files in the folder is greater than the
                        %number of requested samples from this folder.

                        if Nfiles_local>=Nsamples_want(Idasar,Idd)
                            Ichoose=Igood(randperm(length(Igood),Nsamples_want(Idasar,Idd)));  %%Select number of random files needed.

                        else  %%if we are requesting more samples than files, sample with replacement.
                           % Ichoose=Igood(randperm(Nfiles_local));
                            %NN=ceil(Nsamples_want(Idasar,Idd)/Nfiles_local);
                            %Ichoose=repmat(Ichoose,1,NN);Ichoose=Ichoose(:);
                            %Ichoose=Ichoose(1:Nsamples_want(Idasar,Idd));

                            Ichoose=Igood(randi(Nfiles_local,1,Nsamples_want(Idasar,Idd)));
                        end

                       % disp(char(fnames(Ichoose,:)));
                       % pause
                        for Ifile=1:length(Ichoose)
                            eval(sprintf('!cp %s/D%i.dir/%s %s/Unsupervised_database.dir',dir_string,Idd,fnames(Ichoose(Ifile),:),database_folder));
                        end
                    end
                    %char(unique(data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5)))'
                end %Idd
            end

        end
    end
end

%%%Check results
fnames_final=dir([database_folder '/Unsupervised_database.dir/*mat']);
fprintf('There are %i files in database\n',length(fnames_final));
letters=zeros(size(fnames_final,1),1);
for I=1:size(fnames_final,1)
     letters(I)=str2num(fnames_final(I).name(end-4));
end
auto_frac=sum(letters==0)/length(letters);
fprintf('Manual fraction is %6.4f\n',1-auto_frac);

