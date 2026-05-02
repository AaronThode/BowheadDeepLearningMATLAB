
%master_UMAP_3D_detector_review.m


close all
clear all
addpath ..
addpath .

%%%Data review parameters
zlimm_want=[-5 5];  %%%Restrict zaxis when selecting samples
color_label='ICI';  %%How to label colors in 3D scattering plot.

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
%display_call_classifications=false;

dataset_chc='auto';
force_UMAP_recompute=false;
force_labels_recompute=false;

hours_to_exclude_recent_edits=24;  %exclude editing previously edited
                    %   samples if they were made less than XX hours agao
display_manual=false;  %If true, plot spectrogram images of known manual calls
display_auto=true;  %If true, plot spectrogram images of known manual calls
display_NTV=false;


%%%UMAP parameters
addpath ../../../umapAndEppFileExchange_v4_6/umap
UMAP_dim=3;   %Dimension of UMAP to compute
n_neighbors=15;
min_dist=0.1;
save_template=false;


%[Database_dir,procdata_basedir,gitpath] = setUpDatabasePaths;
[latent_space_dir,image_dir,reviewer_initials,manual_file,gsi_dir] = setUpDatabasePaths;

switch dataset_chc
    case 'manual'
        dir_names={[Database_dir 'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir'], ...
            [Database_dir 'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir']};

        images_dir{1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];
       
    case 'auto'
        %Original result, with samples not centered in time...
        %images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir'];
        %images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];

        %%%Baseline
        %dir_names={[Database_dir '/LD32/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/']};
        %dir_names={'../../../Bowhead_DL_Project/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/'};
        %Centered result
        %images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir'];
        %images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir'];

        %%%Revised centered images with everything labeled properly
        clear dir_names
        dir_names={[latent_space_dir 'Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir/']};
        images_dir{1,1}=[image_dir 'Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir'];
        images_dir{1,2}=[image_dir 'Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir'];

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
    %  features is a structure where every field is a vector with same
    %  number of elements as data.x.
    %  The call type is stored in several ways:
    %       data.feature.type_org are the original labels before review.
    %           Always has full classification labels, never alterable by
    %           reviewers.
    %       data.feature.type     are the labels after review (cleaned data set)
    %           Always has full classification labels.
     %       gcf().UserData.CData:  subset of samples being plotted.
    %   Related structures:
    %       data.date_adjusted:   datetime of when data.feature.type was
    %                           altered
    %       data.reviewer:  e.g.. 'AT', initials of reviewer.


    if force_labels_recompute || ~isfield(data.features,'type')
        disp('Adding feature vectors to MAT file before continuing...');
        Npp=size(x,1);
        for II=1:Npp

            if rem(II,100)==0,fprintf('%6.2f percent done\n', 100*II/Npp);end
            fname=data.original_filenames{II};

            try
                if strcmp(dataset_chc,'manual')
                    imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,fname),'features');
                else
                    if strcmp(fname(end-4),'0')
                        imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname),'features');
                    else
                        imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,fname),'features');
                    end
                end
            catch
                fprintf('Could not load %s...\n',fname);
            end
            if II==1
                feature_names=fieldnames(imgdata.features);
                for Ifeature=1:length(feature_names)
                    data.features.(feature_names{Ifeature})=ones(Npp,1);
                end
                data.features.type=ones(Npp,1);
            end

            for Ifeature=1:length(feature_names)
                try
                    data.features.(feature_names{Ifeature})(II)=imgdata.features.(feature_names{Ifeature});
                catch
                    data.features.(feature_names{Ifeature})(II)=-1;
                end
                data.features.type(II)=str2double(extract(fname,28));
            end
        end %%II
        data.features.type_org=data.features.type;
        data.date_adjusted=repmat(datetime('now'),length(data.features.type),1);
        data.reviewer=repmat("XX",length(data.features.type),1);

        save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data")
    end  %Advanced labels

    data.features.iscall=(data.features.type>0);
    

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

    x_norm=(x-mean(x))./std(x);
    x_color=data.features.(color_label);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%Plot all detections with UI controls%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%Optional flip to try to get better view of data...
    x_norm=-x_norm;
    %ud=scatter3_limits_with_azel_edits(x_norm,x_color,[31 -81],zlimm_want);
    myfig=scatter3_GUI_rotate_transparency_filter(x_norm,data.features,data.date_adjusted,[78 90],zlimm_want); colormap jet

    %myfig=gcf;

    disp('Select rotation check and rotate figure');
    drawnow;

    initial_azi=0;
    initial_el=-5;
    alpha_value=0.2;
    %create_gif=input('Enter 1 to create a rotating GIF, hit return otherwise...\n');
    % create_gif=[];
    % ifempty(create_gif)
    %     titstr=sprintf('%s_%s_UMAP%idim.gif',dataset_chc,color_label,UMAP_dim);
    %     GIF_movie_demo(x(Itype,:),x_color(Itype),alpha_value,titstr,initial_azi,initial_el);
    % end


   % list = {'Select and edit subsamples','Do nearest-neighbor analysis','Quit'};
    %[indx,tf] = listdlg('ListString',list, ...
    %    'PromptString','Adjust view(rotation,features, filtering) and then select option:', ...
    %    'SelectionMode','single','InitialValue',1);

   
    operation_chc= input('Adjust view(rotation,features, filtering) and then type ''1'' to review, ''2'' for nearest neighbor analysis...   ');
    if isempty(operation_chc)
        continue
    end

    if ~exist('display_sample','var')
        operation_chc=true;
    end

    notready=true;
    while operation_chc & notready
        select_and_display_samples_from_UMAP_display;

        notready=input('Enter 1 to select more points .....  ');

    end

    cd(mydir)
end