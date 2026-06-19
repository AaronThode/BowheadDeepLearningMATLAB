%%%%master_list_edits_in_latent_file.m
%%% Run this script in the same folder with the file name
close all
clear all

file_name='~/Downloads/latent_embeddings_3d_auto_MATLAB.mat';

data=load(file_name);
reviewer='AH';


I_changed=find(data.features.type~=data.features.type_org);
fprintf('%i images have changed in this file...\n',reviewer,length(I_changed));

I_reviewed=contains(data.reviewer,reviewer);
fprintf('%s has edited %i images...\n',reviewer,sum(I_reviewed));

I_both=find(data.features.type(I_reviewed)~=data.features.type_org(I_reviewed));
fprintf('%s has changed %i labels...\n',reviewer,length(I_both));
