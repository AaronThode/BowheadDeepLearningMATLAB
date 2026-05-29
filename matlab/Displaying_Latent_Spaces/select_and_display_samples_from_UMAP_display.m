%%%select_and_display_samples_from_UMAP_display.m

figure(myfig);
ud=myfig.UserData;
Xt=ud.Xt;  %%Figure might have been rotated after selection, so need to update...
Igood=ud.Igood;  %%%Points visible on screen (survived filtering)

%%%Check that length of Igood matches points on screen
if (length(Igood)~=length(ud.h.CData)) | (length(Igood) ~= length(ud.h.XData))
    keyboard
end

tmp=ginput(2);
tmp(:,3)=str2num(ud.edtZ.String)';
Icluster=find(Xt(Igood,1)>min(tmp(:,1))&Xt(Igood,1)<max(tmp(:,1)) ...
    &Xt(Igood,2)>min(tmp(:,2)) &Xt(Igood,2)<max(tmp(:,2)) ...
    &Xt(Igood,3)>min(tmp(:,3)) &Xt(Igood,3)<max(tmp(:,3)));

Icluster=Igood(Icluster);  %Index now relates fo full data set

%%%Don''t count recent edits...
N_all_subsamples=length(Icluster);
Inot_recent=find(data.date_adjusted(Icluster)<ignore_edits_more_recent_than);  %exclude editing previously edited

Icluster=Icluster(Inot_recent);  %Index now relates fo full data set

temp_fnames=data.original_filenames(Icluster);
temp_type=data.features.type(Icluster);  %%Note that edited calls will be used here...

temp_Imanual=find(temp_type>0);
temp_Iauto=find(temp_type==0);

N_manual=length(temp_Imanual);
N_unmarked=length(temp_Iauto);
fprintf('Out of %i original detections %i have not already been edited... \n\t Of those remaining there are %i manual calls and %i unmarked signals in this sample \n', ...
    N_all_subsamples,length(Icluster),N_manual,N_unmarked);
pause(0.5)
%Display manual examples
hh=ud.ax;
hh.Title.String=sprintf('%i originals, %i Samples editable, %i manual, %i auto', ...
    N_all_subsamples,length(Icluster),N_manual,N_unmarked);
hh.Title.FontWeight="bold";
hh.Title.FontSize=14;
drawnow
%Ncalls=min([30 length(Icluster)]);
%Iwant=(randperm(length(Icluster),Ncalls));

Ichanged_manual=[];
if display_manual& ~isempty(temp_Imanual)
    [data.features.type, Ichanged_manual]=make_tile_spectrograms("Manual",temp_fnames(temp_Imanual), ...
        data.features.type,Icluster(temp_Imanual),dataset_chc,images_dir,manual_file,gsi_dir,display_NTV);
end

Ichanged_auto=[];
if display_auto & ~isempty(temp_Iauto)
    [data.features.type, Ichanged_auto]=make_tile_spectrograms("Auto",temp_fnames(temp_Iauto), ...
        data.features.type,Icluster(temp_Iauto),dataset_chc,images_dir,manual_file,gsi_dir,display_NTV);
    
end
% close(2:length(get(0).Children))

figgs = findall(0,'Type','figure');   % includes hidden handles
for f = figgs.'
    if ~contains(f.Name,'Scatter3')
        close(f);
    end
end
clear figgs f

%%%Ireviewed are indicies that have been reviewed,
%%% Ichanged are indicies that have been changed...

try
    Ireviewed=unique([Icluster(temp_Imanual); Icluster(temp_Iauto)]);

catch
    Ireviewed=unique([Icluster(temp_Imanual) Icluster(temp_Iauto)]);

end
%Ichanged=unique([Ichanged_manual Ichanged_auto]);
data.date_adjusted(Ireviewed)=datetime('now');
data.reviewer(Ireviewed)=reviewer_initials;

%%%Change type and iscall features
ud.features.type=data.features.type;
data.features.iscall=double(data.features.type>0);
ud.features.iscall=data.features.iscall;

figure(myfig);
set(myfig,'UserData',ud);

%%%Update the complete features vector
data.features_org.type(Ifull_dataset_index)=data.features.type;
data.date_adjusted_org(Ifull_dataset_index)=data.date_adjusted;
data.reviewer_org(Ifull_dataset_index,:)=data.reviewer;
data.features_org.iscall(Ifull_dataset_index)=data.features.iscall;

%%%Update the internal 'Igood' variable stored in UserData.

fhandle=myfig.UserData.edtFeature1.Callback;
fhandle(myfig.UserData.edtFeature1);

%%%Update info on edited samples
former_manual=(data.features.type_org>0 & ~data.features.iscall);
former_auto=(data.features.type_org==0 & data.features.iscall);
fprintf('Call samples removed: %i, Call samples added: %i, total changes: %i\n\n', ...
    sum(former_manual),sum(former_auto),sum(data.features.type~=data.features.type_org));