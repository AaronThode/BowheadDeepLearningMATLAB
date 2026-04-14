%%%select_and_display_samples_from_UMAP_display.m

if ~exist('display_sample','var')
    display_sample=true;
end

Xt=gcf().UserData.Xt;


notready=true;
while display_sample & notready
    Xt=gcf().UserData.Xt;

    tmp=ginput(2);
    tmp(:,3)=str2num(gcf().UserData.edtZ.String)';
    Icluster=find(Xt(:,1)>min(tmp(:,1))&Xt(:,1)<max(tmp(:,1)) ...
        &Xt(:,2)>min(tmp(:,2)) &Xt(:,2)<max(tmp(:,2)) ...
        &Xt(:,3)>min(tmp(:,3)) &Xt(:,3)<max(tmp(:,3)));

    temp_fnames=data.original_filenames(Icluster);
    temp_type=type(Icluster);

    temp_Imanual=find(temp_type>0);
    temp_Iauto=find(temp_type==0);

    N_manual=length(temp_Imanual);
    N_unmarked=length(temp_Iauto);
    fprintf('Out of %i detections there are %i manual calls and %i unmarked signals in this sample \n', ...
        length(Icluster),N_manual,N_unmarked);

    %Display manual examples
    hh=gcf().UserData.ax;
    hh.Title.String=sprintf('%i Samples in range, %i manual, %i auto', ...
        length(Icluster),N_manual,N_unmarked);
    hh.Title.FontWeight="bold";
    hh.Title.FontSize=14;

    %Ncalls=min([30 length(Icluster)]);
    %Iwant=(randperm(length(Icluster),Ncalls));

    if display_manual
        make_tile_spectrograms("Manual",temp_Imanual,temp_fnames,dataset_chc,images_dir,display_NTV);
    end
    make_tile_spectrograms("Auto",temp_Iauto,temp_fnames,dataset_chc,images_dir,display_NTV);


    notready=input('Enter 1 to make another selection:');
    close(3:length(get(0).Children))
end