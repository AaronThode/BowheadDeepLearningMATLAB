function [latent_space_dir,image_dir,gitpath,gsi_dir] = setUpDatabasePaths
% set up paths depending on what system I'm using
% latent_space_dir: base directory where all network weights and latent
%   space samples are stored
% image_dir:  directory where unsupervised image databases are stored.
% gsi_dir:    directory where the audio files are stored...
% gitpath: directory of GIT repo sio_research


gsi_dir=[];image_dir=[];latent_space_dir=[];gitpath=[];
success=false;

[~,hostname] = system('hostname');
[~,user_name]=system('whoami');
switch hostname(1:end-1)

    case 'ishmael.ucsd.edu'

        if strcmpi(deblank(user_name),'thode')
            fprintf('Using direct drive on AaronThode''s ishmael account\n')
            gitpath = '/Users/thode/Desktop/ThodeLab';
            latent_space_dir = '/Users/oboulais/Public/Bowhead_DL_Project/';
            success=true;
        end
    otherwise
        fprintf('Using Aaron''s local laptop\n')
        gitpath = '/Users/thode/Desktop/ThodeLab';

        %%%Check if external drive.  If exists use it.
        if exist('/Volumes/Thode_AI_Working_Disk/','dir')==7
            disp('Using external Thode_AI_Working_Disk for latent space and images')
            latent_space_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Networks_And_LatentSpaceRuns.dir/LD32/';
            image_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/BCB_Whale_Datasets/';

        else
            disp('Using external Thode_AI_Working_Disk for latent space...CANNOT review images...')
            latent_space_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Bowhead_DL_Project/LD32/';

        end
        success=true;
end

if ~success
    error('No valid database paths found');
    return
end


%