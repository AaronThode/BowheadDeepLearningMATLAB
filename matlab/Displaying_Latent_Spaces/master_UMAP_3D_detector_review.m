
%master_UMAP_3D_detector_review.m


close all
clear all
addpath ..
addpath .

%%%Data review parameters
zlimm_want=[-5 5];  %%%Restrict zaxis when selecting samples
color_label='iscall';  %%How to label colors in 3D scattering plot.

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

ignore_edits_more_recent_than=datetime(2026,5,18,19,0,0);  %exclude editing previously edited
%   samples if they were made more recently than this
%   time
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
        data.features.score=zeros(size(data.features.type));  %Score is a feature!
        data.reviewer=repmat("XX",length(data.features.type),1);

        data.features=orderfields(data.features);

        %%%Put iscall last so it is default color label in plot...
        fieldname=fieldnames(data.features);
        J_iscall=find(contains(fieldname,'iscall'));
        data.features=orderfields(data.features,[1:(J_iscall-1) (J_iscall+1):length(fieldname) J_iscall]);
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
     myfig=scatter3_GUI_rotate_transparency_filter(x_norm,data.features,[78 90],zlimm_want); colormap jet
    
    disp('Select rotation check and rotate figure');
    drawnow;

    initial_azi=78;
    initial_el=90;
    alpha_value=0.2;
    %create_gif=input('Enter 1 to create a rotating GIF, hit return otherwise...\n');
     create_gif=[];
     if ~isempty(create_gif)
         titstr=sprintf('%s_%s_UMAP%idim.gif',dataset_chc,color_label,UMAP_dim);
        GIF_movie_demo(x_norm,data.features.iscall,alpha_value,titstr,initial_azi,initial_el);
     end


    group = "Updates";
    pref = "Conversion";
    tit = "Choose Operation";
    quest = "After adjusting figure, choose an option:";
    pbtns = ["Edit","Nearest Neighbor Compute","Print","Save","Quit"];

    notready=true;
    while notready
        [operation_chc,tf] = uigetpref(group,pref,tit,quest,pbtns);
        switch operation_chc
            case 'print'
                ud=myfig.UserData;
                fig_print=figure;
                ax_print=axes;
                newobj=copyobj(ud.h,ax_print);

                Hax = findobj(myfig,'type','axes');
                
                set(ax_print,'XLim',Hax.XLim);
                set(ax_print,'YLim',Hax.YLim);
                set(ax_print,'ZLim',Hax.ZLim);
                
                colormap jet; grid on;
                hh=colorbar; 
                set(hh.Label,"String",ud.selectedFeatureField, "FontSize",12,"FontWeight","bold")
                clim([ min(ud.CData)  max(ud.CData)]);
                
                xlabel('UMAP 1');ylabel('UMAP 2');


                titstr= sprintf('latent_embeddings_%id_%s_MATLAB',UMAP_dim,dataset_chc);
                title(sprintf('%s, Azimuth: %s, Elevation: %s, Filtering %s between %s and %s', ...
                    titstr,ud.edtAz.String,ud.edtEl.String, ...
                    ud.selectedFilterField,ud.edtFeature1.String,ud.edtFeature2.String),'interp','none');
                set(gca,'FontWeight','bold','FontSize',14);
                hh = findobj(myfig,'type','colorbar');
                set(hh.Label,"String",ud.selectedFeatureField, "FontSize",12,"FontWeight","bold")
                printname=sprintf('%s_Azi%s_El%s_Feat.%s_Filt.%s.jpg',titstr, ...
                    ud.edtAz.String,ud.edtEl.String, ...
                    ud.selectedFeatureField,ud.selectedFilterField);

                orient landscape
                print('-djpeg','-r300',printname);
            case 'edit'
                disp('Editing...')
                select_and_display_samples_from_UMAP_display;

            case 'nearest neighbor compute'
                disp('Computing nearest neighbor....')
                ud=myfig.UserData;
                Igood=ud.Igood;
                Neighbors=20;
                Idx=knnsearch(data.latent_embeddings(Igood,:), ...
                    data.latent_embeddings(Igood,:),'K',Neighbors,'IncludeTies',true,'Distance','euclidean');

                threshold=unique([ 0 0:1/Neighbors:1 1]);
                strrr='kr';strr2='bg';
                figure;
                for I=1:2  %1 is original manual labels, 2 is latest relabel
                    switch I
                        case 1
                            iscall=data.features.type_org>0 & data.features.type_org<12;
                        case 2
                            iscall=data.features.type>0 & data.features.type<12;
                    end

                    score=zeros(length(iscall),1);
                    for Iscore=1:length(Idx)
                        clusster=iscall(Igood(Idx{Iscore}));
                        score(Igood(Iscore))=sum(clusster)/Neighbors;
                    end
                    Icall=iscall(Igood)>0;
                    Ino_call=(iscall(Igood)==0);

                    total_positives_in_dataset=sum(Icall);

                    recall=zeros(1,length(threshold));precision=recall;
                    for Ithresh=1:length(threshold)
                        detected_positives=sum(score(Igood(Icall))>=threshold(Ithresh));
                        false_positives=sum(score(Igood(Ino_call))>=threshold(Ithresh));
                        recall(Ithresh)=detected_positives./total_positives_in_dataset;
                        precision(Ithresh)=detected_positives./(detected_positives+false_positives);

                    end

                  
                    subplot(1,2,1);hold on
                    plot(recall,precision,[strrr(I) '-o']);grid on;hold on
                    xlabel('Recall');ylabel('Precision');
                    xlim([0 1]);ylim([0 1]);
                    subplot(1,2,2);hold on
                    plot(1-recall,1-precision,[strrr(I) '-o']);grid on;
                    xlabel('Miss fraction');ylabel('False discovery rate (Fraction of calls that are not calls)');
                    xlim([0 1]);ylim([0 1]);

                    %%%Estimate precision for realistic data set
                    R1=sum(Ino_call)/sum(Icall);
                    R2=7.8;  %Ratio of false hits to manual count in full dataset
                    precision_estimated=precision.*R1./(R2.*(1-precision)+R1.*precision);
                    subplot(1,2,1)
                    plot(recall,precision_estimated,[strr2(I) '-o']);
                    subplot(1,2,2);
                    plot(1-recall,1-precision_estimated,[strr2(I) '-o']);grid on;hold on
                   

                end %I
                legend('original','estimated bulk: original','edited','estimated bulk: edited');
                title(sprintf('Dataset length: %i samples (%i calls, %i false)',length(Igood),sum(Icall),sum(Ino_call)))
                
                ud.features.score=score;
                
                set(myfig,'UserData',ud);



            case 'save'
                disp('Saving...')
                save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data");
            case 'quit'
                notready=false;

        end
    end
    cd(mydir)
end