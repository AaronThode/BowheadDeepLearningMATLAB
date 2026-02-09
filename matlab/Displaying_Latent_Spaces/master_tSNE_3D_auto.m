%master_tSNE_3D_manual.m
%

close all
clear all

dir_names={'/Volumes/SIO_THODE1/DeepLearningBowheadWhale/TrainedModels.dir/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/', ...
    '/Volumes/SIO_THODE1/DeepLearningBowheadWhale/TrainedModels.dir/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/'};


markersize=5;

for Idir=1:length(dir_names)
    disp(dir_names{Idir})
    mydir=pwd;
    cd([dir_names{Idir} filesep 'MATLAB'])
    data=load('latent_embeddings.mat');
    save_flag=false;

    if ~isfield(data,'x_tsne')
        disp('Recomputing tSNE...')
        data.x_tsne=tsne(data.latent_embeddings,'NumDimensions',3);
        save_flag=true;
        save('latent_embeddings.mat','-struct','data');

    end

    figure(Idir)
    %h(Idir,1)=subplot(1,1,1);

    type=str2double(extract(data.original_filenames,28));
    type(type>0)=1;
    Itype=1:length(type);
    alpha_value=0.1;
    %for J=1:2 %%Split by call type
    % switch J
    %     case 1
    %         Itype=find(type==0);
    %         alpha_value=0.1;
    %         %titstr='automated detections';
    %
    %     case 2
    %         Itype=find(type>0);
    %         %titstr='Manual Whale calls';
    %          alpha_value=0.1;
    % end

    titstr='automated detections and manual whale calls';

    ss(Idir,1)=scatter3(data.x_tsne(Itype,1), data.x_tsne(Itype,2), data.x_tsne(Itype,3), markersize,type(Itype),'filled');
    ss(Idir,1).MarkerEdgeAlpha=alpha_value;
    ss(Idir,1).MarkerFaceAlpha=alpha_value;
    grid on
    axis equal
    colorbar
    colormap jet


    xlabel('Dimension 1');
    ylabel('Dimension 2');
    zlabel('Dimension 3');
    hold on


    Istart=strfind(dir_names{Idir},'LD');

    title([dir_names{Idir}(Istart(1)+(0:3)) ' ' titstr]);
    %linkaxes([h(Idir,1) h(Idir,2)]);

    % hLink = linkprop(h(Idir,:), {'CameraPosition','CameraUpVector','CameraTarget'});
    clear data
    cd(mydir)
end