function make_tile_spectrograms(FigureName,fnames,dataset_chc,images_dir,plot_NTV)

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
separation_distance=[0 7 7 7*sqrt(3) 14 7*sqrt(7) 21];  %Distance between DASARs in km

fnames=sort(fnames);  %Sort by alphabetical order to group by site and year
Nsamples=length(fnames);

figure(Name=FigureName);set(gcf,'Position',[ 11          60        1745         874  ]);
Iplot=0;
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
    target_time(10:end)='000000';
    tsec=seconds(tabs_call-datetime(target_time));
    tsec_shift=tsec*(head.tdrift/86400);
    tabs_call=tabs_call-duration(0,0,tsec_shift);  %%This makes acoustic time match manually-logged time
    
     
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


    
    ctime_call=posixtime(tabs_call)+8*3600;  %All ctimes are in UTC
    call_event.ctime=manual_logs.manual_data{Isite,Iyear}.localized.ctev_UTC;  %Time of call event
    call_event.ctimes=manual_logs.manual_data{Isite,Iyear}.ind.ctime(:,Iletter);  %Time of call reception at target DASAR
    call_event.flo=manual_logs.manual_data{Isite,Iyear}.ind.flo(:,Iletter);  %Time of call reception at target DASAR
    call_event.fhi=manual_logs.manual_data{Isite,Iyear}.ind.fhi(:,Iletter);  %Time of call reception at target DASAR
    
    call_event.ctimes(isnan(call_event.ctimes))=Inf;

    %%%Note that times in manual logs are in UTC time!
    call_event.tabs=datetime(1970,1,1,-8,0,call_event.ctimes); %-8 converts from UTC time (archive) to local time (GSI WAV)
    call_event.ranges=manual_logs.manual_data{Isite,Iyear}.localized.range;

    %Ctimes in UTC, datetimes (tabs) in local time...
    
    %%%First see if target call is in manual database, and check if
    %%%frequencies overlap
    type_color='w';
    minn=abs(tabs_call-call_event.tabs);
    if any(minn<duration(0,0,1.5))
        Ibest=find(minn<duration(0,0,1.5));
        %Check that frequencies match...
        if (imgdata.features.fpeak>=call_event.flo(Ibest)) & (imgdata.features.fpeak<=call_event.fhi(Ibest))
            type_color='g'; %Frequency match as well
        else
            type_color='y';
        end
    else
        min(minn) 
    end

    temp=abs(abs(ctime_call-call_event.ctime)-call_event.ranges(Iletter,:)'/1.5);
    Iloc_candidate=find(temp<5);
    match_score=zeros(2,length(Iloc_candidate));
    for K=1:length(Iloc_candidate)
        II=Iloc_candidate(K);
        call_cand.flows=manual_logs.manual_data{Isite,Iyear}.ind.flo(II,:);
        call_cand.fhighs=manual_logs.manual_data{Isite,Iyear}.ind.fhi(II,:);
        call_cand.SNR=manual_logs.manual_data{Isite,Iyear}.ind.stndb(II,:);
        call_cand.ctimes=manual_logs.manual_data{Isite,Iyear}.ind.ctime(II,:);  %Time of call reception at target DASAR
    
        Isep=abs((1:7)-Iletter)+1;
        possible_matches=abs(call_cand.ctimes-call_cand.ctimes(Iletter))./(1.5*separation_distance(Isep)/1.5);
        frequency_test=call_cand.flows<imgdata.features.fpeak & call_cand.fhighs > imgdata.features.fpeak;
        match_score(1,K)=sum((possible_matches<1)); 
        match_score(2,K)=sum(frequency_test&(possible_matches<1));  %Don't count myself because it will be a NaN
    end
    if isempty(max(match_score(1,:)))||isnan(max(match_score(1,:)))
        match_score_all(1)=0;
    else
        match_score_all(1)=max(match_score(1,:));
    end

    if isempty(max(match_score(2,:)))||isnan(max(match_score(2,:)))
        match_score_all(2)=0;
    else
        match_score_all(2)=max(match_score(2,:));
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

    for Iloc_candidate=1:Nplots_per_sample

        if Iloc_candidate==1
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
        text(0.1,450,int2str(Iplot),'color',type_color,'fontsize',12);  %Plot number
        
        text(2.0,450,fnames{(JJ)}(end-4),'color',type_color,'fontsize',12);  %Call type
        
        text(0.1,-20,sprintf('%3.1f dB',outputs.SNR),'color','k','fontsize',8); %SNR
        text(0.1,400,sprintf('%3.1f s',seconds(min(minn))),'color',type_color,'fontsize',8);%closest manual call
        text(0.1,380,sprintf('%i %i mtch',match_score_all(1),match_score_all(2)),'color',type_color,'fontsize',8);%closest manual call
       
        if ~isempty(outputs.duration)
             text(0.1,-50,sprintf('%3.1f s',imgdata.features.duration1),'color','k','fontsize',8);
            % imgdata
        end
    end %I
end %JJ