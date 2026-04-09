function make_tile_spectrograms(FigureName,Iindex,fnames,dataset_chc,images_dir)

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
    disp(fnames{Iindex(JJ)})
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

    imagesc(TT,FF,imgdata.SNR_gram);%colorbar;
    ylim([0 500]);
    axis xy
    set(gca,'fontweight','bold','fontsize',14)
    title(fnames{Iindex(JJ)}(1:22),'FontSize',8);
    text(0.1,450,fnames{Iindex(JJ)}(end-4),'color','w','fontsize',12)
    if rem(JJ,10)~=1
        set(gca,'ytick',[]);
    else
        ylabel('Hz')
    end
    if JJ<21
        set(gca,'xtick',[]);
    else
        xlabel('Time (sec)')
    end
end %J