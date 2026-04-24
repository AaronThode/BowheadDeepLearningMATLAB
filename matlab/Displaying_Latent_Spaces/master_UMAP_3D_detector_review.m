
%master_UMAP_3D_detector_review.m


close all
clear all
addpath ..
addpath .

dataset_chc='auto';
force_UMAP_recompute=false;
force_Labels_recompute=true;
color_label='type';  %%How to label colors in 3D scattering. 
% fpeak: 
%         tpeak:  
%     duration2:  %Duration estimated by peak-picking image (not accurate)
%           SNR:  
%          fmin:  
%          fmax:  
%     duration1:  %%%Duration computed from original event detector
%        dB_RMS:  
%     magnitude:  
%           ICI:  


%%%UMAP parameters
addpath ../../../umapAndEppFileExchange_v4_6/umap
UMAP_dim=3;   %Dimension of UMAP to compute
n_neighbors=15;
min_dist=0.1;
save_template=false;

%%%Data review parameters
zlimm_want=[0.2 0.4];  %%%Restrict zaxis when selecting samples
display_manual=true;
display_NTV=false;
display_call_classifications=false;

[Database_dir,procdata_basedir,gitpath] = setUpDatabasePaths;
switch dataset_chc

    case 'manual'
        dir_names={[Database_dir 'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir'], ...
            [Database_dir 'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir']};

        images_dir{1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];
        %images_dir{2}=images_dir{1};

        Ntypes=length(images_dir);
    case 'auto'
         %Original result, with samples not centered in time...
        %images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir'];
        %images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];

        %%%Baseline
        dir_names={[Database_dir '/LD32/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/']};
        %dir_names={'../../../Bowhead_DL_Project/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/'};
         %Centered result
        images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir'];
        images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir'];

        %%%Revised with everything labeled properly
        clear dir_names
        Database_dir='../../../Bowhead_DL_Project/';
        dir_names={[Database_dir '/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir/']};
        %Centered result

        Image_database_dir='/Volumes/Maui2025';
        %Image_database_dir='/Volumes/Bowhead_DL_Project/';
        images_dir{1,1}=[Image_database_dir '/BCB_Whale_Datasets/Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir'];
        images_dir{1,2}=[Image_database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir'];

       

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
        save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data");
    else
        x=data.(field_want);
    end

    %%If stored MAT file does not have features stored in convenient form,
    %%add it!
    if force_Labels_recompute || ~isfield(data,'type')
        disp('Adding feature vectors to MAT file before continuing...');
        Npp=size(x,1);
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

            if II==1
                feature_names=fieldnames(imgdata.features);
                for Ifeature=1:length(feature_names)
                    data.(feature_names{Ifeature})=ones(Npp,1);
                end
                    data.type=ones(Npp,1);
            end

            for Ifeature=1:length(feature_names)
                try
                    data.(feature_names{Ifeature})(II)=imgdata.features.(feature_names{Ifeature});
                catch
                    data.(feature_names{Ifeature})(II)=-1;
                end
                data.type(II)=str2double(extract(fname,28));
            end
        end %%II
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

    if isfield(data,'type')
        disp('Taking call type from data object (preferred)')
        type=data.type;
    else
        disp('Reading call type from file name')
        type=str2double(extract(data.original_filenames,28));
        data.type=type;
    end


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

                %%%Detection problem only
                if ~display_call_classifications
                    type(type>0)=1;
                    data.type(data.type>0)=1;
                end
                switch Jcat
                    case 1
                        Itype=find(type>0);
                        titstr='Manual calls';


                    case 2
                        Itype=find(type==0);
                        titstr='Generic detection';

                end
                %Itype=find(type==0);
                alpha_value=0.02;

        end
        % h(Idir,J)=subplot(1,2,J);
        x_norm=(x-mean(x))./std(x);

        x_color=data.(color_label);
        % switch color_label
        %     case 'fpeak'
        %         x_color=data.PeakFrequency;
        %     case 'tpeak'
        %         x_color=data.PeakTime;
        %         case 'duration1'
        %         x_color=data.duration1;
        %     case 'type'
        %         x_color=data.type;
        % end


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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%Plot all detections with UI controls%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%Optional flip to try to get better view of data...
    x_norm=-x_norm;
    %ud=scatter3_limits_with_azel_edits(x_norm,x_color,[31 -81],zlimm_want);
    ud=scatter3_limits_with_azel_edits(x_norm,x_color,[132 50],zlimm_want); colormap jet
    
   

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
    select_and_display_samples_from_UMAP_display;


    clear data
    cd(mydir)
end