%%%select_and_display_samples_from_UMAP_display.m

if ~exist('display_sample','var')
    display_sample=true;
end

notready=true;
while display_sample & notready

    drawnow
    figure(myfig);
    ud=myfig.UserData;
    Xt=ud.Xt;
    Igood=ud.Igood;  %%%Points visible on screen (survived filtering)
    tmp=ginput(2);
    tmp(:,3)=str2num(gcf().UserData.edtZ.String)';
    Icluster=find(Xt(Igood,1)>min(tmp(:,1))&Xt(Igood,1)<max(tmp(:,1)) ...
        &Xt(Igood,2)>min(tmp(:,2)) &Xt(Igood,2)<max(tmp(:,2)) ...
        &Xt(Igood,3)>min(tmp(:,3)) &Xt(Igood,3)<max(tmp(:,3)));

    Icluster=Igood(Icluster);  %Index now relates fo full data set

    temp_fnames=data.original_filenames(Icluster);
    temp_type=data.features.type(Icluster);  %%Note that edited calls will be used here...

    temp_Imanual=find(temp_type>0);
    temp_Iauto=find(temp_type==0);

    N_manual=length(temp_Imanual);
    N_unmarked=length(temp_Iauto);
    fprintf('Out of %i detections there are %i manual calls and %i unmarked signals in this sample \n', ...
        length(Icluster),N_manual,N_unmarked);

    %Display manual examples
    hh=gcf().UserData.ax;
    hh.Title.String=sprintf('%i Samples in range, %i manual, %i auto', ...
        length(Icluster),N_manual,N_unmarked);
    hh.Title.FontWeight="bold";
    hh.Title.FontSize=14;
    drawnow
    %Ncalls=min([30 length(Icluster)]);
    %Iwant=(randperm(length(Icluster),Ncalls));

    Ichanged_manual=[];
    if display_manual& ~isempty(temp_Imanual)
        [data.features.type, Ichanged_manual]=make_tile_spectrograms("Manual",temp_fnames(temp_Imanual), ...
            data.features.type,Icluster(temp_Imanual),dataset_chc,images_dir,display_NTV);
    end

    Ichanged_auto=[];
    if display_auto & ~isempty(temp_Iauto)
        [data.features.type, Ichanged_auto]=make_tile_spectrograms("Auto",temp_fnames(temp_Iauto), ...
            data.features.type,Icluster(temp_Iauto),dataset_chc,images_dir,display_NTV);
    end
    close(2:length(get(0).Children))

    figgs = findall(0,'Type','figure');   % includes hidden handles
    for f = figgs.'
        if f.Number ~= 1
            close(f)
        end
    end

    Ichanged=unique([Ichanged_manual Ichanged_auto]);
    data.date_adjusted(Ichanged)=datetime('now');
    data.reviewer(Ichanged)=reviewer_initials;
    %%%Change UMAP color scheme...
    ud.features.type=data.features.type;
    data.features.iscall=double(data.features.type>0);
    ud.features.iscall=data.features.iscall;

    figure(1);
    % hhh=gcf();
    % hhh.UserData.features.type=data.features.type;
    % hhh.UserData.features.iscall=double(ud.features.iscall);


    if strcmpi(ud.selectedFeatureField,'iscall')
        Igood=find(ud.features.iscall>0);
        ud.Igood=Igood;
        data.Igood=Igood;
        set(myfig,'UserData',ud);

        fhandle=myfig.UserData.edtFeature1.Callback;
        fhandle(myfig.UserData.edtFeature1);
    else
        set(myfig,'UserData',ud);
    end
    %  warning off
    %   set(ud.h,'XData',ud.Xt(ud.Igood,1),'YData',ud.Xt(ud.Igood,2),'ZData',ud.Xt(ud.Igood,3),'CData',double(ud.CData(ud.Igood)));
    % % set(ud.h,'CData',double(ud.CData(ud.Igood)));
    % warning on


    set(myfig,'UserData',ud);
    notready=input('Enter 1 to make another selection:');

end