%master_UMAP_3D_detector_review.m


close all
clear all
addpath ..
addpath .

dataset_chc='auto';
force_UMAP_recompute=false;
UMAP_dim=3;   %Dimension of UMAP to compute
color_label='type';  %%How to label colors in 3D scattering.  'PeakFrequency' or 'type','PeakTime'
advanced_labels=false;
addpath ../../../umapAndEppFileExchange_v4_6/umap
n_neighbors=15;
min_dist=0.1;
save_template=false;
%n_components=3;

[Database_dir,procdata_basedir,gitpath] = setUpDatabasePaths;
switch dataset_chc

    case 'manual'
        dir_names={[Database_dir 'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir'], ...
            [Database_dir 'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir']};

        images_dir{1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];
        %images_dir{2}=images_dir{1};

        Ntypes=length(images_dir);
    case 'auto'
        % dir_names={[Database_dir '/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir']};
        %dir_names={[Database_dir '/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir']};
        dir_names={[Database_dir '/LD32/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/']};
        %dir_names={'../../../Bowhead_DL_Project/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/'};

        %Original result
        %images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir'];
        %images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];

        %Centered result
        images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir'];
        images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir'];


        Ntypes=2;
end

for Idir=1:length(dir_names)
    disp(dir_names{Idir})
    mydir=pwd;


    %%MATLAB UMAP processing
    cd([dir_names{Idir} filesep 'MATLAB'])

    if  force_UMAP_recompute
        file_want=sprintf('latent_embeddings.mat');
    else
        file_want=sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc);
    end

    fpath = fullfile(pwd, file_want);    % current folder + filename
    if isfile(fpath)                  % or: exist(fpath,'file')==2
        data = load(fpath);
    else
        error('File "%s" not found in current folder: %s', file_want, pwd);
    end


    %%%Compute UMAP results...
    field_want=sprintf('umap_embeddings_%id',UMAP_dim);

    if force_UMAP_recompute
        [x, umap, clusterIds, extras]= ...
            run_umap(double(data.latent_embeddings),'n_components', UMAP_dim,'min_dist',min_dist,'n_neighbors',n_neighbors,'verbose','text');
        data.(field_want)=x;
        %save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data");
    else
        x=data.(field_want);
    end

    %%If frequency information not available, load from SNR_gram
    if advanced_labels & ~isfield(data,'PeakFrequency')
        Npp=size(x,1);
        data.PeakFrequency=ones(Npp,1);
        data.PeakTime=ones(Npp,1);
        for II=1:Npp
            if rem(II,100)==0,fprintf('%6.2f percent done\n', 100*II/Npp);end
            fname=data.original_filenames{II};
            if strcmp(dataset_chc,'manual')
                imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,fname));
            else
                if strcmp(fname(end-4),'0')
                    imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname));
                else
                    imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,fname));
                end
            end

            [outputs]=extract_features_from_SNRgram(imgdata.dT,imgdata.dF,imgdata.SNR_gram);
            data.PeakFrequency(II)=outputs.Fmax;
            data.PeakTime(II)=outputs.Tmax;


        end %%II
        %save(sprintf('umap_embeddings_%id.mat',UMAP_dim),'PeakTime','PeakFrequency',"-append");
        save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data")
    end  %Advanced labels

    if UMAP_dim==5
        [coeff,score,latent,tsquared,explained] = pca(x,'NumComponents',3);
        %coeff: projection of original axes onto new orthogonal axes (5
        %by 3 matrix)
        %
        % score:  translation of each data point into the new PCA
        % coordinates
        %  latent: variance of each column of score
        %  explained: percent varience explained by PCA component.
        %  Used to judge which components to keep
        x=score;
    end

    cd(mydir)

    type=str2double(extract(data.original_filenames,28));


    %%%Create overview plot to help identify where to search for calls.
    for Jcat=1:Ntypes %%Split by call type
        figure(Idir)

        switch dataset_chc
            case 'manual'
                initial_azi=45;
                initial_el=-25;

                switch Jcat
                    case 1
                        Itype=find(type<4);
                        Itype=find(type<4 | type==7);

                        titstr='Upsweeps, downsweeps, and constant tones';
                        alpha_value=0.3;
                    case 2
                        Itype=find(type==7);
                        titstr='Complex Calls';
                        alpha_value=0.5;
                end
            case 'auto'
                initial_azi=0;
                initial_el=-5;

                %type(type>0)=1;
                switch Jcat
                    case 1
                        Itype=find(type>0);
                        titstr='Manual calls';


                    case 2
                        Itype=find(type==0);
                        titstr='Generic detection';

                end
                %Itype=find(type==0);
                alpha_value=0.3;

        end
        % h(Idir,J)=subplot(1,2,J);
        x_norm=(x-mean(x))./std(x);


        switch color_label
            case 'PeakFrequency'
                x_color=data.PeakFrequency;
            case 'PeakTime'
                x_color=data.PeakTime;
            case 'type'
                x_color=type;
        end


        %%%Plot both individually
        %for K=1:2
        h(Idir,Jcat)=subplot(1,2,Jcat);
        %if Jcat==1
        ss(Idir,Jcat)=scatter3(x(Itype,1), x(Itype,2), x(Itype,3), 3,x_color(Itype),'filled');
        % else
        % ss(Idir,K)=scatter3(x_norm(Itype,1), x_norm(Itype,2), x_norm(Itype,3), 3,x_color(Itype),'filled');

        % end
        ss(Idir,Jcat).MarkerEdgeAlpha=alpha_value;
        ss(Idir,Jcat).MarkerFaceAlpha=alpha_value;
        grid on
        axis equal
        colorbar
        %if J==1
        colormap jet
        % else
        %    colormap gray
        % end

        xlabel('Dimension 1');
        ylabel('Dimension 2');
        zlabel('Dimension 3');
        title([dir_names{Idir}(1:4) ' ' titstr]);
        if Jcat==2
            linkaxes([h(Idir,1) h(Idir,2)]);
            linkprop([h(Idir,1) h(Idir,2)],'View');
        end
        % end

        %hLink = linkprop(h(Idir,:), {'CameraPosition','CameraUpVector','CameraTarget'});

    end %Jcat

    %%%Plot all detections with UI controls
    scatter3_limits_with_azel_edits(x_norm,x_color);
    colormap jet

    myfig=gcf;

    disp('Select rotation check and rotate figure');
    drawnow;

    %create_gif=input('Enter 1 to create a rotating GIF, hit return otherwise...\n');
    % create_gif=[];
    % if ~isempty(create_gif)
    %     titstr=sprintf('%s_%s_UMAP%idim.gif',dataset_chc,color_label,UMAP_dim);
    %     GIF_movie_demo(x(Itype,:),x_color(Itype),alpha_value,titstr,initial_azi,initial_el);
    % end


    display_sample= input('Switch to transform view, rotate and press 1 when ready...');
    if isempty(display_sample)
        continue
    end
    Xt=gcf().UserData.Xt;


    notready=true;
    while display_sample && notready
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
        hh.Title.String=sprintf('%i Samples in range',length(Icluster));
        hh.Title.FontWeight="bold";
        hh.Title.FontSize=14;

        %Ncalls=min([30 length(Icluster)]);
        %Iwant=(randperm(length(Icluster),Ncalls));

        make_tile_spectrograms("Manual",temp_Imanual,temp_fnames,dataset_chc,images_dir);
        make_tile_spectrograms("Auto",temp_Iauto,temp_fnames,dataset_chc,images_dir);

        
        notready=input('Enter 1 to make another selection:');
        close(3:length(get(0).Children))
    end
    %end %J


    clear data
    cd(mydir)
end