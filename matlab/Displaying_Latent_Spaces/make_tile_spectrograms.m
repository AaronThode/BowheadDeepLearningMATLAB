function make_tile_spectrograms(FigureName,Iindex,fnames,dataset_chc,images_dir,plot_NTV)

if ~exist('plot_NTV','var')
    plot_NTV=true;
end

Nplots_per_sample=1;
if plot_NTV
    Nplots_per_sample=2;
end

Nsamples=length(Iindex);

figure(Name=FigureName);set(gcf,'Position',[ 11          60        1745         874  ]);
Iplot=0;
for JJ=1:Nsamples
    Iplot=Iplot+1;
    if Iplot>30
        Iplot=1;
        figure(Name=FigureName);set(gcf,'Position',[ 11          60        1745         874  ]);
    end

    subplot(3,10,Iplot)

    fprintf('%s \n',fnames{Iindex(JJ)})
    % type(Itype(Icluster(Iwant(JJ))));  %Should be same type as
    % in file name

    if strcmp(dataset_chc,'manual')
        imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,fnames{Iwant(JJ)}));
    else
        try
            if strcmp(fnames{Iindex(JJ)}(end-4),'0')
                load_name=sprintf('%s%s%s',images_dir{1},filesep,fnames{Iindex(JJ)});
                imgdata=load(load_name);
            else
                load_name=sprintf('%s%s%s',images_dir{2},filesep,fnames{Iindex(JJ)});
                imgdata=load(load_name);
            end
        catch
            fprintf('%s not in directory...\n',load_name);
            continue
        end
    end


    FF=imgdata.dF*(0:size(imgdata.SNR_gram,1));
    TT=imgdata.dT*(0:size(imgdata.SNR_gram,2));

    for I=1:Nplots_per_sample

        if I==1
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
        title(fnames{Iindex(JJ)}(1:22),'FontSize',8);
        text(0.1,450,fnames{Iindex(JJ)}(end-4),'color','w','fontsize',12)
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
        text(0.1,20,sprintf('%3.1f dB',outputs.SNR),'color','w','fontsize',8);
        if ~isempty(outputs.duration)
             text(0.1,50,sprintf('%3.1f s',outputs.duration),'color','w','fontsize',8);
        end
    end %I
end %JJ