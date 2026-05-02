function [latent_space_dir,image_dir,review_initials,gitpath,gsi_dir] = setUpDatabasePaths
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

 if strcmpi(deblank(user_name),'thode')
     review_initials='AT';
 elseif strcmpi(deblank(user_name),'oboulais')
     review_initials='OB';
 end



switch hostname(1:end-1)

    case 'ishmael.ucsd.edu'

        if strcmpi(deblank(user_name),'thode')
            fprintf('Using direct drive on AaronThode''s ishmael account\n')
            gitpath = '/Users/thode/Desktop/ThodeLab';
            latent_space_dir = '/Users/oboulais/Public/Bowhead_DL_Project/';

            %%%Check if GSI data available
            gsi_dir='~/mnt/jonah3/Shared/Data';
            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            success=true;
        end
    otherwise
        fprintf('Using Aaron''s local laptop\n')
        gitpath = '/Users/thode/Desktop/ThodeLab';

        %%%Check if GSI data available
        gsi_dir='/Volumes/Shared/Data/';

        if ~exist(gsi_dir,'dir')==7
            disp('No GSI directory found, cannot link DASARs or play sounds');
        end

        %%%Check if external drive.  If exists use it.
        if exist('/Volumes/Thode_AI_Working_Disk/','dir')==7
            disp('Using external Thode_AI_Working_Disk for latent space and images')
            latent_space_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Networks_And_LatentSpaceRuns.dir/LD32/';
            image_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/BCB_Whale_Datasets/';

        else
            disp('Using internal laptop storage for latent space.')
            latent_space_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Bowhead_DL_Project/LD32/';

            %%%Check if external server mounted for images
            if exist('/Volumes/Bowhead_DL_Project/','dir')==7
                image_dir='/Volumes/Bowhead_DL_Project/BCB_Whale_Datasets/';
            else
                disp('WARNING!  CANNOT DISPLAY IMAGES....')

            end
        end
        success=true;
end

if ~success
    error('No valid database paths found');
    return
end


%