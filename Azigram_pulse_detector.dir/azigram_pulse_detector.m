function [output, TT,FF, azi, B, ItoE]=azigram_pulse_detector(x,Fs,Nfft,ovlap,params)
%function [output, TT,FF, azi, B, ItoE]=azigram_pulse_detector(x,Fs,Nfft,ovlap,params)

%%%Inputs:x:  cell matrix with x{Idasar} being a N by 3 column matrix from
%%%         DASAR Idasar
%%%%Parameters
%params.tabs_start:  %%Start time of x in datenumber.
%params.threshold=5*4;  %Hz
%params.f_transition;  %Frequency at which DASAR p and v go out of phase.
%params.vertical_line=round(params.vertical_line/dF);
%params.min_TBW=round(params.min_TBW/(dT*dF));
%params.brefa=vector the same size as x;
%params.frange
%params.debug
%params.a_grid, da
%params.threshold
%params.time_chunk=10*60;  %How much data to read into memory at a
%           time--solves clock drift problem too
% params.morph_choice='raw','open','close'
%params.base_ltr=ID of DASARs
%  params.min_perc_det: For a given frequency bin, what fraction of
%           mask detection duration has value of '1'?  Used to compute
%           frequency bandwidth more accurately.

%%%%Output:
 %%% TT,FF, time and frequency vectors
 %%% azi{Istation}, B{Istation},ItoE{Istation}: azigram and spectrogram cell arrays
params_temp=params;
Nstation=length(x);  %How many DASARs are participating?

%

%%%Determine how big a chunk to import

params.time_chunk=min([params.time_chunk length(x{1})/Fs]);
%Nt=floor(params.time_chunk/dT);
%Npt=floor(1+((length(x{1})/Nfft)-1)/(1-ovlap));
Iwindow=unique([1:(params.time_chunk*Fs):length(x{1}) length(x{1})]);  %Index of chunks

if params.time_chunk==length(x{1})/Fs  %%%Chunk is whole window
    Iwindow=[1 length(x{1})+1];
end

%%%Index of detections
Icount=0;

dI=Nfft*(1-ovlap);  %Samples per time unit
dF=Fs/Nfft;
params.vertical_line=round(params.vertical_line/dF);
params.min_TBW=round(params.min_TBW/(dI*dF));

SE=strel('line',params.vertical_line,90);  %%%closing object

%%%Loop through time windows
for It=1:(length(Iwindow)-1)
    % for It=88:(length(Iwindow)-1)
    fprintf('\n\nProcessing chunk %i of %i\n',It,length(Iwindow)-1);
    It_index=Iwindow(It):(Iwindow(It+1)-1);
    
    
    for Ichan=1:Nstation
        params_temp.brefa=params.brefa(Ichan);
        [TT,FF,azi{Ichan},B{Ichan},~,Ix{Ichan},Iy{Ichan}]=compute_directional_metrics ...
            (x{Ichan}(:,It_index),{'Directionality','ItoERatio'},Fs,Nfft,ovlap,params_temp, ...
            'gsi',[false false ]);
        
        ItoE{Ichan}=azi{Ichan}{2};  %Transport ratio
        azi{Ichan}=azi{Ichan}{1};
    end
    
    
    %%%%Compute azigram difference
    %%%Convert parameter values into pixels
    dF=FF(2)-FF(1);
    dT=TT(2)-TT(1);  %How many seconds per spectrogram bin
    
    
    
    %%%%If the image is very long, break up into smaller pieces for this loop
    %%%%Crude time synchronization....
    % Try to time-align spectrograms to best of ability..
    
    if params.debug.input
        figure(5);
        %plot(temp,'k');grid on;
        plot_azigram_overview(TT,FF,B,azi,params)
        drawnow;
    end
    
    [B,azi,ItoE,Ix,Iy,delta_T(It,:),TT]=crude_time_synchronize_images(B,azi,ItoE,Ix,Iy,TT);
    
    %%%%%%%%Plot spectrograms and azigrams
    if params.debug.input
        figure(6);
        %plot(temp,'k');grid on;
        plot_azigram_overview(TT,FF,B,azi,params)
        drawnow;
        keyboard
    end
    %%%Create a series of binary images by azimuthal sector
    
    Iff=find(FF>=params.frange(1)&FF<=params.frange(2));
    for I=1:Nstation
        bw_all{I}=false(length(Iff),size(azi{1},2),length(params.a_grid));
        for Ia=1:length(params.a_grid)
            %Make binary images
            %fprintf('Searching between %6.2f and %6.2f degrees\n',params.a_grid(Ia)-params.da/2,params.a_grid(Ia)+params.da/2);
            bw_all{I}(:,:,Ia)=(azi{I}(Iff,:)>=(params.a_grid(Ia)-params.da/2))&(azi{I}(Iff,:)<=(params.a_grid(Ia)+params.da/2));
        end
    end
    
    %optionss={'raw','open','close'};
    optionss={'raw'};
    
    %Ia=9;  %Demo for 10 seconds into DASAR A, 1/18 Tako City
    %Ia=13;  %Demo for 10 seconds into DASAR A, 1/18 Tako City
    for J=1:length(optionss)
        err.(optionss{J})=zeros(length(params.a_grid),length(params.a_grid),length(TT),'uint16');
    end
    disp('Starting a_grid');
    
    %%%Loop through all combinations of azimuthal sectors for first two
    %%%DASARs in list.
    %%%  comb.raw:  combined binary images with no morph processing
    %%%  comb.clearn.noclose: combined binary images after morph opening
    %%%  (remove small objects)
    
    for Ia=1:length(params.a_grid)
        fprintf('Image processing azi %i degrees: %6.2f percent done\n',params.a_grid(Ia),100*Ia/length(params.a_grid));
        
        %%%Remove small objects and store original result
        
        bw{1}=squeeze(bw_all{1}(:,:,Ia));
        
        %%%keep image for future masking...
        %bw_all{1}(:,:,Ia)=bw{1};
        
        for Ia1=1:length(params.a_grid)
            %    for Ia1=2:6
            
            bw{2}=squeeze(bw_all{2}(:,:,Ia1));
            
            %bw_all{2}(:,:,Ia1)=bw{2};
            
            comb.raw=bw{1}&bw{2};
            %comb.open=bwareaopen(comb.raw,params.min_TBW);
            %comb.close=imclose(comb.open,SE);
            
            for J=1:length(optionss)
                err.(optionss{J})(Ia,Ia1,:)=dF*sum(comb.(optionss{J}));
            end
            
            
            if params.debug.morph
                if any(dF*sum(comb.(params.morph_choice))>params.threshold)
                    %Needs TT FF Iff azi azi{2} bw bw1 comb dBspread  B  B{2}  params
                    %Ia Ia 1
                    plot_morphological_debug;
                    
                    pause
                end
            end
            
        end  %Ia1
    end %Ia
    
    detect=err.(params.morph_choice);
    
    %detection units (azimuth1, azimuth2, time)
    detect=[detect(end,:,:); detect; detect(1,:,:)]; %Takes care of cyclical permutation around north
    detect=[detect(:,end,:) detect detect(:,1,:)];
    
    %%Augment bw_all images by cyclical permuation
    for I=1:Nstation
        bw_all{I}(:,:,2:(end+1))=bw_all{I};
        bw_all{I}(:,:,1)=bw_all{I}(:,:,end);  %%%Beginning and end are the same
        bw_all{I}(:,:,end+1)=bw_all{I}(:,:,2); %I had made a mistake here, June 27.  bw_all is now two angles larger
    end
    
    %%%%%%%%For every combination of angles, search for common
    %%%%%%%%  pulses.
    
    %for Ia=2:(length(params.a_grid)-1)
    for Ia=2:(length(params.a_grid)+1)  %The 2 is because we'd artifically bounded the detect function with
        % a wrap around, so only examine from the second to next-to-last
        % detect vallues.
        
        for Ia1=2:(length(params.a_grid)+1)
            det=squeeze(detect(Ia,Ia1,:));  %Now just a time series with amplitude in Hz
            %fprintf('Angle 1: %6.2f Angle 2: %6.2f\n',params.a_grid(Ia-1),params.a_grid(Ia1-1));
            
            Itest=find(det>params.threshold);
            if isempty(Itest)  %%%If no pulses found, move on
                continue
            end
            
            %%%Identify distinct pulses by looking for time gaps
            % Merge dips between two close peaks
            % Works for peaks separated by sep_len or less
            seplen = floor(params.sep_len/(dI/Fs)); % separated by at least
            if seplen<1; seplen=1; end % require at least one sample sep
            
            Ibound=unique([0; find(diff(Itest)>seplen); length(Itest)]); % % % % SEPARATION WIDTH
            
            %Ibound=unique([1 ; find(diff(Itest)>seplen); length(Itest)]);
            %fprintf('There are %i pulses in this segment\n',length(Ibound));
            
            if params.debug.detect
                figure(20);clf;plot(det,'s-');grid on;hold on; title(sprintf('%i %i: %i segments',Ia,Ia1,length(Ibound)-1));
                det_ind=1:length(det);
                plot(det_ind(Itest),det(Itest),'ro');hold on;
                line([1 length(det)],params.threshold*[1 1])
            end
            for Ipulse=1:(length(Ibound)-1)  %%%For each pulse...
                if rem(Ipulse,100)==0,fprintf('%6.2f percent done\n',100*Ipulse/length(Ibound));end
                II=(Ibound(Ipulse)+1):(Ibound(Ipulse+1));
                %II=(Ibound(Ipulse)):(Ibound(Ipulse+1));
                
                index=Itest(II);  %%%indicies in det that define pulse..defines duration of pulse
                
                % Emma add buffer to either side
                db = round(params.snips.buffer/2*Fs/dI);
                %i1 = max(min([index(1)-db, index(1)]), 1); 
                %i2 = min(max([index(end)+db, index(end)]),length(det));
     
                if params.debug.detect
                    plot(det_ind(index),det(index),'gx');pause
                end
                %%%% TT(index) is the time from the
                %%%% start of the azigram/specgram
                
                %%%%%Check that duration is not too long--i.e. a motorboat
                if dT*length(index)>params.max_time
                    fprintf('Ouch!  Pulse %i is %6.2f sec long.\n',Ipulse,dT*length(index));
                    continue
                end
                
                %%%Check that a higher score does not exist in an adjacent bin.
                %%%% I looked at both area of peak and height.  Problem with
                %%%% area is that a peak might split into two in an adjacent
                %%%% bin (index differs between bins).
                %%%  The peak height is a more robust check of "best"
                %%%  viewpoint.
                %score=sum(det(index));
                score=max(det(index));
                local_peak_flag=true;
                
                %%%Mask is just the relevant section of the bw image
                if params.debug.mask_construction
                    mask=false(size(bw_all{1}(:,:,1)));
                    index_all=1:size(bw_all{1},2);
                    index2=index;
                else
                    index_all=index;
                    index2=1:length(index_all);  %%%How wide (duration) is the binary mask?
                    mask=false(size(bw_all{1}(:,index_all,1)));
                    
                end
                
                %%%Check if we are near the "best" combination of Ia and Ia1
                % for a particular peak
                for JJ=[-1 0 1]
                    for KK=[-1 0 1]
                       
                        % Check within the separation length for stronger
                        % pulses (just get rid of weaker for now) Emma
                        addtofront = [];%(index(1)-seplen:index(1)-1).';
                        addtofront(addtofront<1) = [];
                        addtoback = [];%(index(end)+1:index(end)+seplen).';
                        addtoback(addtoback>size(detect,3)) = [];
                        lateral_index = [addtofront; index; addtoback];
                        
                        %temp=sum(detect(Ia+JJ,Ia1+KK,index),3);
                        temp=max(detect(Ia+JJ,Ia1+KK,lateral_index));
                        %fprintf('Ia: %i Ia1: %i, JJ:%i KK:%i, score: %6.2f\n',Ia, Ia1, JJ,KK,temp);
                        
                        local_peak_flag=local_peak_flag&(score>=temp);  %If this turns false, never turns back
                    
                        % Merge two lateral loops Emma
                        % check if flag went false first
                        recall_detect(Ia+JJ,Ia1+KK,index_all)=detect(Ia+JJ,Ia1+KK,index_all);
                        
                        %%%if this detection is not larger than
                        %%%surroundings, break out of loop
                        if ~local_peak_flag
                           break; 
                        end
                        if params.debug.mask_construction
                            comb.raw=(bw_all{1}(:,:,Ia+JJ)&bw_all{2}(:,:,Ia1+KK));
                            comb.raw(:,1:(index(1)-1))=false;
                            comb.raw(:,(index(end)+1):end)=false;
                            
                            
                            %%%Comfirm that I retreive original detection
                            %%%result
                            %                             if I1==Ia&I2==Ia1
                            %                                 fprintf('detect value: %s, reconstructed value: %s\n',mat2str(squeeze(detect(Ia,Ia1,index_all))),mat2str(dF*(sum(comb.raw))));
                            %                                   keyboard
                            %                             end
                        else  %this is not a debug display
                            comb.raw=(bw_all{1}(:,index_all,Ia+JJ)&bw_all{2}(:,index_all,Ia1+KK));
                            
                        end
                        %comb.open=bwareaopen(comb.raw,params.min_TBW);
                        %comb.close=imclose(comb.open,SE);
                        
                        % if params.debug.mask_construction & JJ+KK==2
                        %    plot_mask_construction;
                        %end
                        
                        if params.debug.mask_construction
                            figure(21)
                            subplot(2,1,1)
                            imagesc(mask)
                            title(sprintf('JJ %i KK %i',JJ,KK));
                        end
                        
                        mask=(mask |comb.(params.morph_choice)); %%Grow mask with contributions from all sides
                        
                         if params.debug.mask_construction
                            figure(21)
                            subplot(2,1,2)
                            imagesc(mask)
                            pause;
                        end
                        %%%Clear detection function so won't trigger in future
                        %%% June 27, 2019
                        %%%  Aaron removed...
                        %detect(Ia+JJ,Ia1+KK,index_all)=0;

                        % Emma zero laterals
                        %detect(Ia+JJ,Ia1+KK,lateral_index)=0;
                    end
                     % Emma break from loop if flag false
                     if ~local_peak_flag
                           break; 
                     end
                end
                if ~local_peak_flag  %%Haven't hit the local maximum in error surface
                    % close(20)
                    continue
                end
                
                Icount=Icount+1;  %Increment
                
                
                %%%%%Use all stations to compute bearings and position...
                for I=1:Nstation  
                    if params.debug.mask_construction
                        plot_debug_mask_construction;
                        pause
                    end
                    
                    %%%%Weighted_median
                    
                    [output.azi.wm.med(I,Icount),output.azi.wm.iqr(I,Icount)]=get_weighted_median(mask(:,index2), ...
                        azi{I}(Iff,index),B{I}(Iff,index),params.da);
                    
                    
                    %%%Compute total energy in two channels
                    output.azi.avg(I,Icount) = atan2d(sum(sum(mask(:,index2).*real(Ix{I}(Iff,index)))), ...
                        sum(sum(mask(:,index2).*real(Iy{I}(Iff,index)))));
                    output.azi.avg(I,Icount)=bnorm(params.brefa(I)+output.azi.avg(I,Icount));
                    
                    temp=mask(:,index2).*(10.^(0.1*B{I}(Iff,index)));
                    [Pmax,Itempf]=max(max(temp,[],2),[],1); 
                    output.power.peakPSD(I,Icount)=10*log10(Pmax);
                    output.freq.peak(I,Icount)=FF(Iff(Itempf));
                    fsum = sum(temp,2);
                    [~,Iftemp] = max(fsum,[],1);
                    [~,locs] = findpeaks(fsum, 'MinPeakProminence', std(fsum)*10^(5/10));
                    output.freq.npeak(I,Icount) = length(locs);
                    output.freq.sumpeak(I,Icount) = FF(Iff(Iftemp));
                    [~,Itempt]=max(sum(temp,1));
                    seconds_from_start=TT(index(Itempt))+delta_T(It,I)+It_index(1)/Fs;
                    
                    output.trel_peak(I,Icount)=seconds_from_start;
                    output.tabs_peak(I,Icount)=datenum(0,0,0,0,0,seconds_from_start)+params.tabs_start;
                   
                    output.azi.peak(I,Icount)=azi{I}(Iff(Itempf),index(Itempt));
                    
                    temp=temp(temp~=0);
                    output.power.SEL(I,Icount)=10*log10(sum(temp));
                    output.power.PSD.med(I,Icount)=10*log10(median(temp));
                    output.power.PSD.iqr(I,Icount)=iqr(10*log10(temp));
                    
                    temp=mask(:,index2).*ItoE{I}(Iff,index);
                    temp=temp(temp~=0);
                    output.ItoE.med(I,Icount)=median(temp);
                    output.ItoE.iqr(I,Icount)=iqr(temp);
                    
                    %output.tabs(1,Icount)=datenum(0,0,0,0,0,TT(index(1))+0.5*dT+It_index(1)/Fs)+params.tabs_start;  %%%Need start of time chunk
                    %%% index(1) is start of detection within this time
                    %%% chunk
                    %%% It_index(1) is start of the time window
                    %%% this result matches with time series, but not
                    %%% azigram window--need to add delta_T(It,Istation)
                    %%% back.
                    seconds_from_start=TT(index(1))+delta_T(It,I)+It_index(1)/Fs;
                    output.tabs(I,Icount)=datenum(0,0,0,0,0,seconds_from_start)+params.tabs_start;
                    output.trel(I,Icount)=seconds_from_start;
                    output.irel(I,Icount) = round(seconds_from_start*Fs);  %index number
                 
                end  %Nstation
                
                %Ifr=(Iff(any(mask(:,index2),2))); % % original line,
                %requiring just 1 detection
                min_det_len = floor(params.min_perc_det*length(index2));
                Ifr=(Iff(sum(mask(:,index2),2)>min_det_len)); 
                if ~isempty(Ifr)
                    output.freq.min(1,Icount)=FF(Ifr(1));
                    output.freq.max(1,Icount)=FF(Ifr(end));
                else
                   output.freq.min(1,Icount) = 0;
                   output.freq.max(1,Icount) = 0;
                end
                output.duration.max(1,Icount)=dT*length(index);
                output.BW.initial(1,Icount)=double(score);
                output.BW.final(1,Icount)=dF*max(sum(mask));
                output.TWP.max(1,Icount)=dT*sum(double(det(index)));
                
                %output.BW.max_close(1,Icount)=dF*double(max(squeeze(err(Ia,Ia1,index,3))));
                %output.TWP.max_close(1,Icount)=dF*dT*sum(double(squeeze(err(Ia,Ia1,index,3))));
                
                %%%To speed up, zero out detections that are adjacent here...
                %detect(Ia+(-1:1),Ia1+(-1:1),index)=0;
            end %Ipulse
            
            
        end  %Ia1
    end %Ia
    
end  %It

output.delta_T=delta_T;

%%%Are there any detections in this window at all?
if ~isfield(output,'trel_peak')
    fprintf('No detections found\n');
    return
end
%%%Sort output by increasing time, not by azimuth secto
[~,Isort]=sort(output.trel_peak(1,:));
% output.trel
% output.azi.wm.med

output=sort_structs(output,Isort);
output.Isort=Isort;  %Preserves original processing order
% output.trel
% output.azi.wm.med


