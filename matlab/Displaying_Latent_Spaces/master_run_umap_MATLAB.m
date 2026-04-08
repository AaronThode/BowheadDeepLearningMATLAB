
clear
close all
addpath ../../../umapAndEppFileExchange_v4_6/umap
n_neighbors=15;
min_dist=0.1;
save_template=false;
n_components=3;
dataset_chc='auto';

%%%Load original UMAP from python output...
%base_file='umap_embeddings_3d_auto';
%original_python=load(['../../../Bowhead_DL_Project/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/UMAP/' base_file '.mat']);
 
base_file='latent_embeddings';
dir_name='../../../Bowhead_DL_Project/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/MATLAB/';
original_python=load([ dir_name filesep base_file '.mat']);



[umap_embeddings_3d, umap, clusterIds, extras]= ...
    run_umap(double(original_python.latent_embeddings),'n_components', n_components,'min_dist',min_dist,'n_neighbors',n_neighbors,'verbose','text');


%%%Save template for future use...
if save_template
    disp('Saving template...')
    template_file=[base_file '_UMAP_template.mat'];
    save(template_file,'umap');
    disp('Template saved');

    reduction2=run_umap(double(original_python.latent_embeddings(1:20,:)),'verbose','text','template_file',template_file);

end

if ~isfield(original_python,'umap_embeddings_3d')
    original_python.umap_embeddings_3d=umap_embeddings_3d;
    file_want=sprintf('latent_embeddings_%id_%s_MATLAB.mat',n_components,dataset_chc);

    save([ dir_name filesep file_want ],"-struct","original_python");


end

alpha_value=0.1;

subplot(1,2,1)
x=umap_embeddings_3d;
ss=scatter3(x(:,1),x(:,2),x(:,3),3,'k','filled');
ss.MarkerEdgeAlpha=alpha_value;
ss.MarkerFaceAlpha=alpha_value;
title('MATLAB'); grid on

subplot(1,2,2)
x=original_python.umap_embeddings_3d;
ss=scatter3(x(:,1),x(:,2),x(:,3),3,'k','filled');
ss.MarkerEdgeAlpha=alpha_value;
ss.MarkerFaceAlpha=alpha_value;
title('PYTHON');grid on

%umap_embeddings_3d=single(reduction);
%save('umap_embeddings_3d_auto_MATLAB.mat','umap_embeddings_3d');
