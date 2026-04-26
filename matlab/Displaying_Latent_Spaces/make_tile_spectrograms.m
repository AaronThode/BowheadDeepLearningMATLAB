function make_tile_spectrograms(FigureName,fnames,dataset_chc,images_dir,plot_NTV)

imgdata_min_freq=25; %Minimum frequency
spectrogram_window_length=3;
persistent manual_logs head_info
manual_file='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Shell_Manual_Results';
manual_file=[manual_file filesep 'All_manual_results.mat'];

if isempty(manual_logs)
    disp('make_tile_spectrogram: Loading manual data...');
    manual_logs=load(manual_file);
end

if isempty(head_info)
    disp('make_tile_spectrogram: Loading head info...');
    head_info=load('/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Software/matlab/GSI_header_table.mat');
end
if ~exist('plot_NTV','var')
    plot_NTV=true;
end

Nplots_per_sample=1;
if plot_NTV
    Nplots_per_sample=2;
end


%%%Prepare for searching for calls in adjacent DASARs
spectrogram_duration=3;  %seconds
separation_distance=[0 7 7 7*sqrt(3) 14 7*sqrt(7) 21];  %Distance between DASARs in km

fnames=sort(fnames);  %Sort by alphabetical order to group by site and year
Nsamples=length(fnames);

figure(Name=FigureName);set(gcf,'Position',[ 11          60        1745         874  ]);
Iplot=0;
Ibest_this_DASAR=zeros(1,Nsamples);
for JJ=1:Nsamples
    fprintf('%s \n',fnames{JJ})

    % type(Itype(Icluster(Iwant(JJ))));  %Should be same type as
    % in file name

    %%%Extract information about time/location from filename
    Isite=str2num(fnames{JJ}(2));
    Iyear=str2num(fnames{JJ}(3:4))-7;
    Iletter=double(fnames{JJ}(5))-double('A')+1;


    %%%Times in filenames are not corrected for clock drift (i.e. they can
    %%%be used to look up acoustic data, but cannot be compared to
    %%%time-corrected logs without adjustment...
    head=get_GSI_head_info(head_info,fnames{JJ}(3:4),fnames{JJ}(2),fnames{JJ}(5));
    target_time=fnames{JJ}(8:22);
    tabs_call=datetime(fnames{JJ}(8:22));  %%%The filename is the time of the midpoint of the detection.
    target_time(10:end)='000000';  %Time at beginning of file

    %Last term in next line needed because file name is midpoint of detection, not
    %beginning

    tsec_call=seconds(tabs_call-datetime(target_time))-spectrogram_duration/2;
    tsec_shift=tsec_call*(head.tdrift/86400);
    tabs_call=tabs_call-duration(0,0,tsec_shift);  %%This makes acoustic time match manually-logged time
    tsec_call=tsec_call-tsec_shift;

    ctime_call=posixtime(tabs_call);  %c-time of call event in local time zone (same as filename)

    if strcmp(dataset_chc,'manual')
        imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fnames{(JJ)}));
    else
        try
            if strcmp(fnames{(JJ)}(end-4),'0')
                load_name=sprintf('%s%s%s',images_dir{1},filesep,fnames{(JJ)});
                imgdata=load(load_name);
            else
                load_name=sprintf('%s%s%s',images_dir{2},filesep,fnames{(JJ)});
                imgdata=load(load_name);
            end
        catch
            fprintf('%s not in directory...\n',load_name);
            continue
        end
    end

    imgdata.features.fpeak=imgdata_min_freq+imgdata.features.fpeak;

    %Process manual data, converting c-times to local time zone (-8 hours)
    %%%Note that ctimes in manual logs 'ind' variable are in UTC time, so need to be shifted back 8 hours!

    time_zone_offset=median(manual_logs.manual_data{Isite,Iyear}.localized.ctev-manual_logs.manual_data{Isite,Iyear}.localized.ctev_UTC);
    call_events.ctime=manual_logs.manual_data{Isite,Iyear}.localized.ctev;  %Time of all call events
    call_events.ctimes=manual_logs.manual_data{Isite,Iyear}.ind.ctime+time_zone_offset;  %Time of call reception at target DASAR
    call_events.flo=manual_logs.manual_data{Isite,Iyear}.ind.flo(:,Iletter);  %Time of call reception at target DASAR
    call_events.fhi=manual_logs.manual_data{Isite,Iyear}.ind.fhi(:,Iletter);  %Time of call reception at target DASAR
    call_events.ctimes(isnan(call_events.ctimes))=Inf;

    call_events.tabs=datetime(1970,1,1,0,0,call_events.ctimes);
    call_events.ranges=manual_logs.manual_data{Isite,Iyear}.localized.range;
    call_events.delta_r=manual_logs.manual_data{Isite,Iyear}.localized.axmajor;
    %%%First see if target call is in manual database, and check if
    %%%frequencies overlap
    type_color='w';
    %minn=abs(tabs_call-call_events.tabs);
    minn=abs(ctime_call-call_events.ctimes(:,Iletter));

    %if any(minn<duration(0,0,1.5))
    if any(minn<=1.5*spectrogram_window_length/2)
        [~,Ibest_this_DASAR(JJ)]=min(minn);
        %Check that frequencies match...
        if (imgdata.features.fpeak>=call_events.flo(Ibest_this_DASAR(JJ))) & (imgdata.features.fpeak<=call_events.fhi(Ibest_this_DASAR(JJ)))
            type_color='g'; %Frequency match as well
        else
            type_color='y';
        end



    else
        min(minn)
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%Plot all manual calls to check that things match the input image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Locate localized manual call closest in possible time to this
    %  detection...

    predicted_ctimes=call_events.ctime+call_events.ranges'/1.5;  %predicted manual call time at site
    temp=abs(ctime_call-predicted_ctimes(:,Iletter));
    Iloc_candidate=find(temp<spectrogram_window_length*5/3);
    match_score=zeros(4,length(Iloc_candidate));
    for K=1:length(Iloc_candidate)
        II=Iloc_candidate(K);
        call_cand.flows=manual_logs.manual_data{Isite,Iyear}.ind.flo(II,:);
        call_cand.fhighs=manual_logs.manual_data{Isite,Iyear}.ind.fhi(II,:);
        call_cand.SNR=manual_logs.manual_data{Isite,Iyear}.ind.stndb(II,:);
        call_cand.ctimes=manual_logs.manual_data{Isite,Iyear}.ind.ctime(II,:)+time_zone_offset;  %Time of call reception at target DASAR

        %%%Test 1, are any manual calls within possible timing range
         Ndasar=sum(~isnan(call_events.ctimes(II,:)));
       
        Isep=abs((1:7)-Iletter)+1;
        possible_matches=abs(call_cand.ctimes-ctime_call)./(1.5*separation_distance(Isep)/1.5);
        frequency_test=call_cand.flows<imgdata.features.fpeak & call_cand.fhighs > imgdata.features.fpeak;
        match_score(1,K)=sum((possible_matches<1))./Ndasar;
        match_score(2,K)=sum(frequency_test&(possible_matches<1))./Ndasar;  %Don't count myself because it will be a NaN

        %%%Test 2: use location to estimate timing of other call
       delta_ctime=abs(predicted_ctimes(II,:)-call_events.ctimes(II,:))<spectrogram_window_length*5/3;
        match_score(3,K)=sum(frequency_test&delta_ctime)./Ndasar;

        delta_ctime=abs(predicted_ctimes(II,:)-call_events.ctimes(II,:));
        delta_ctime(isinf(delta_ctime))=0;
        Ndasar=sum(delta_ctime>0);
        match_score(4,K)=1./(sum(delta_ctime,2)./Ndasar);  %RMS value
    end  %K candidate

    match_score_all=[0 0 0 0];
    for K=1:4
        if length(Iloc_candidate)==1
            match_score_all(K)=match_score(K);
        elseif isempty(max(match_score(K,:)))||isnan(max(match_score(K,:)))
            match_score_all(K)=0;
        else
            match_score_all(K)=max(match_score(K,:));
        end
    end
   

    %manual_logs.manual_data{Isite,Iyear}.localized.wctype(Ipossible_manual_call);

    Iplot=Iplot+1;
    if Iplot>30
        Iplot=1;
        figure(Name=FigureName);set(gcf,'Position',[ 11          60        1745         874  ]);
    end

    subplot(3,10,Iplot)


    FF=imgdata.dF*(0:size(imgdata.SNR_gram,1));
    TT=imgdata.dT*(0:size(imgdata.SNR_gram,2));

    for Iplot_index=1:Nplots_per_sample

        if Iplot_index==1
            imagesc(TT,FF,double(imgdata.SNR_gram)/5);colorbar;

        elseif plot_NTV
            Iplot=Iplot+1;
            subplot(3,10,Iplot)
            imagesc(TT,FF,double(imgdata.NTV_gram)/100);
            colorbar;
            clim([0 1]);
        else  %I==2 and not plot_NTV
            continue
        end
        ylim([0 500]);
        axis xy
        set(gca,'fontweight','bold','fontsize',14)
        title(fnames{(JJ)}(1:22),'FontSize',8);
        if rem(Iplot,10)~=1
            set(gca,'ytick',[]);
        else
            ylabel('Hz')
        end
        if Iplot<21
            set(gca,'xtick',[]);
        else
            xlabel('Time (sec)')
        end

        [outputs]=extract_features_from_SNRgram(imgdata.dT,imgdata.dF,imgdata.SNR_gram);
        text(0.1,450,int2str(JJ),'color',type_color,'fontsize',12);  %Plot number

        text(2.0,450,fnames{(JJ)}(end-4),'color',type_color,'fontsize',12);  %Call type

        text(0.1,-20,sprintf('%3.1f dB',outputs.SNR),'color','k','fontsize',8); %SNR
        text(0.1,400,sprintf('%3.1f s',(min(minn))),'color',type_color,'fontsize',8);%closest manual call
        text(0.1,380,sprintf('%3.1f %3.1f %3.1f mtch',match_score_all(1), ...
            match_score_all(2),match_score_all(3)),'color',type_color,'fontsize',8);%closest manual call

        if ~isempty(outputs.duration)
            text(0.1,-50,sprintf('%3.1f s',imgdata.features.duration1),'color','k','fontsize',8);
            % imgdata
        end
    end %I
end %JJ

%%%Plot manual detections closest to this detectio across all DASARS....

JJ=input('Enter a number to see related detections on other DASARS: ');
while ~isempty(JJ)
    %%%Option to plot spectrograms of all linked manual detections
    Isite=str2num(fnames{JJ}(2));
    Iyear=str2num(fnames{JJ}(3:4))-7;
    plot_manual_detection_allDASARs(fnames{JJ},manual_logs.manual_data{Isite,Iyear}.ind,time_zone_offset,head_info,Ibest_this_DASAR(JJ));
    
     if strcmp(dataset_chc,'manual')
        imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fnames{(JJ)}));
    else
        try
            if strcmp(fnames{(JJ)}(end-4),'0')
                load_name=sprintf('%s%s%s',images_dir{1},filesep,fnames{(JJ)});
                imgdata=load(load_name);
            else
                load_name=sprintf('%s%s%s',images_dir{2},filesep,fnames{(JJ)});
                imgdata=load(load_name);
            end
        catch
            fprintf('%s not in directory...\n',load_name);
            continue
        end
    end

    imgdata.features.fpeak=imgdata_min_freq+imgdata.features.fpeak;
    subplot(4,2,8)
    FF=imgdata.dF*(0:size(imgdata.SNR_gram,1));
    TT=imgdata.dT*(0:size(imgdata.SNR_gram,2));
    imagesc(TT,FF,double(imgdata.SNR_gram)/5);colorbar;axis xy;title(fnames{JJ})
    hold on;plot(0.1,imgdata.features.fpeak-imgdata_min_freq,'o','color','w');
    pause;

    JJ=input('Enter a number to see related detections on other DASARS: ');

end
