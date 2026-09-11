%%%%master_evaluate_autoencoder_performance.m

close all
clear all

Neighbors=40;  %Found no effect in using 10 or 40.
min_duration=0.5;
xscale_chc='linear';
overlay_2010_performance=true;

%%%Load data set used to train autoencoder
train_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir/MATLAB';

train_fname=[train_dir filesep 'latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat'];
train=load(train_fname);

%%%Load evaluation data set latent vectors
eval_dir=train_dir;
eval_fname=[eval_dir filesep 'latent_embeddings_3d_eval_8to1_MATLAB_combined.mat'];
eval=load(eval_fname);

disp('Computing nearest neighbor....')

Idx_all=knnsearch(train.latent_embeddings, ...
    eval.latent_embeddings,'K',Neighbors,'IncludeTies',true,'Distance','euclidean');
Nsamples=length(Idx_all);

threshold=unique([ 0 0:1/Neighbors:1 1]);
strr1=':-';strr2='rk';strr3='os';
type_chc={'type_org','type'};


figure;

Ictr=0;
for Itype=1:2
    for Iduration=1:2  %1 is all samples, 2 is restricted duration
        %train.iscall=train.features.type_org>0 & train.features.type_org<12;
        %eval.iscall=eval.features.type_org>0 & eval.features.type_org<12;
        Ictr=Ictr+1;

        train.iscall=train.features.(type_chc{Itype})>0 & train.features.(type_chc{Itype})<12;
        eval.iscall=eval.features.(type_chc{Itype})>0 & eval.features.(type_chc{Itype})<12;
        switch Iduration
            case 1
                Igood=1:Nsamples;
                Idx=Idx_all;

            case 2
                Igood=find(eval.features.duration1>=min_duration);
                Idx=Idx_all(Igood);
                titstr=sprintf('Neighbors: %i, Samples > %3.2f sec',Neighbors,min_duration);
        end
        titstr=sprintf('Neighbors: %i',Neighbors);
        score{Iduration}=zeros(length(eval.iscall(Igood)),1);  %Score is size of test dataset
        idx_max=0;
        for Iscore=1:length(Idx)
            clusster=train.iscall(Idx{Iscore});  %ID (call or not) of nearest neighbors in training set
            score{Iduration}(Iscore)=sum(clusster)/Neighbors;  %%%Fraction of neigbors that are calls.
            idx_max=max([idx_max Idx{Iscore}]);
        end
        idx_max
        Icall=eval.iscall(Igood)>0;
        Ino_call=(eval.iscall(Igood)==0);

        total_positives_in_test=sum(Icall);

        recall=zeros(1,length(threshold));
        precision=recall;
        for Ithresh=1:length(threshold)
            detected_positives=sum(score{Iduration}(Icall)>=threshold(Ithresh));
            false_positives=sum(score{Iduration}(Ino_call)>=threshold(Ithresh));
            recall(Ithresh)=detected_positives./total_positives_in_test;
            precision(Ithresh)=detected_positives./(detected_positives+false_positives);

        end

        subplot(1,2,1);hold on


        % Plot recall vs precision with marker colors set by threshold


        hl(Ictr)=plot(recall,precision,[strr1(Iduration) strr2(Itype)],'LineWidth',2);grid on;hold on
        cmap = jet(length(threshold)); % colormap
        for Iplot=1:length(threshold)
            plot(recall(Iplot),precision(Iplot),[strr1(Iduration) strr3(Itype)], ...
                'Color',cmap(Iplot,:),'MarkerSize',5,'MarkerFaceColor',cmap(Iplot,:));hold on
        end
        colormap(cmap);
        xlabel('Recall');ylabel('Precision');
        xlim([0 1]);ylim([0 1]);
        cb = colorbar('Ticks',linspace(0,1,5),'TickLabels',num2str(round(interp1(linspace(0,1,length(threshold)),threshold,linspace(0,1,5))',3)));
        title(cb,'Threshold');

       
        title(titstr)

        text(0.05,0.95,'a)','fontweight','bold','FontSize',14)

        set(gca,'fontweight','bold','FontSize',14);


        subplot(1,2,2);hold on

        hr(Ictr)=plot(1-recall,1-precision,[strr1(Iduration) strr2(Itype)],'LineWidth',2);grid on;hold on
        cmap = jet(length(threshold)); % colormap
        for Iplot=1:length(threshold)
            plot(1-recall(Iplot),1-precision(Iplot),[strr1(Iduration) strr3(Itype)], ...
                'Color',cmap(Iplot,:),'MarkerSize',5,'MarkerFaceColor',cmap(Iplot,:));hold on
        end
        colormap(cmap);

        xlabel('Miss fraction');ylabel('False discovery rate (Fraction of calls that are not calls)');
        xlim([0 1]);ylim([0 1]);
         cb = colorbar('Ticks',linspace(0,1,5),'TickLabels',num2str(round(interp1(linspace(0,1,length(threshold)),threshold,linspace(0,1,5))',3)));
        title(cb,'Threshold');


        title(titstr)
        %if Iduration==2
        %     legend('All samples',sprintf('Samples greater than %3.2f seconds',min_duration),'location','northeast')
        % end

        if strcmpi(xscale_chc,'log')
            text(1e-4,0.95,'b)','fontweight','bold','FontSize',14)
            xscale log
        else
            text(0.05,0.95,'b)','fontweight','bold','FontSize',14)

        end
        set(gca,'fontweight','bold','FontSize',14);


        %%%Estimate precision for realistic data set
        % R1=sum(Ino_call)/sum(Icall);
        % R2=7.8;  %Ratio of false hits to manual count in full dataset
        % precision_estimated=precision.*R1./(R2.*(1-precision)+R1.*precision);
        % subplot(1,2,1)
        % plot(recall,precision_estimated,[strr2(I) '-o']);
        % subplot(1,2,2);
        % plot(1-recall,1-precision_estimated,[strr2(I) '-o']);grid on;hold on

        fprintf('Dataset length: %i samples (%i calls, %i false)\n',length(eval.iscall(Igood)),sum(Icall),sum(Ino_call))

    end %I
end %Itype
%title(sprintf('%i Neighbors',Neighbors))

 
% legend(hl,'All samples (original)',sprintf('> %3.2f seconds',min_duration),'All samples (reviewed)',sprintf('> %3.2f seconds (reviewed)',min_duration),'location',' southwest')

if overlay_2010_performance
    old_data=load('~/Publications/Arctic_work/DeepLearningAutoEncoder_BowheadDetection.dir/OldNeuralNetPerformance.dir/OldResults_for_Comparison.mat');
    yearly_samples=[38773 151511 93427 148967]';
    % Extract the ithth element from every vector in the cell array ygrid


    for Igrid=1:length(old_data.xgrid)
        y10 = [cellfun(@(v) v(Igrid), old_data.ygrid(:))];

        numerator=yearly_samples.*y10;
        Isgood=~isnan(numerator);
        old_FDR(Igrid)=sum(numerator(Isgood))/sum(yearly_samples(Isgood));
    end
    
    subplot(1,2,1)
    hl(end+1)=plot(1-old_data.xgrid,1-old_FDR,'--','linewidth',2,'color','k');
    subplot(1,2,2)
    hr(end+1)=plot(old_data.xgrid,old_FDR,'--','linewidth',2,'color','k');
    legend(hl, 'All samples (original)',sprintf('> %3.2f seconds',min_duration),'All samples (reviewed)',sprintf('> %3.2f seconds (reviewed)',min_duration),'Original 2010 network','location',' best')
    legend(hr, 'All samples (original)',sprintf('> %3.2f seconds',min_duration),'All samples (reviewed)',sprintf('> %3.2f seconds (reviewed)',min_duration),'Original 2010 network','location',' best')

else
    legend(hl, 'All samples (original)',sprintf('> %3.2f seconds',min_duration),'All samples (reviewed)',sprintf('> %3.2f seconds (reviewed)',min_duration),'location',' best')
    legend(hr, 'All samples (original)',sprintf('> %3.2f seconds',min_duration),'All samples (reviewed)',sprintf('> %3.2f seconds (reviewed)',min_duration),'location',' best')

end


orient landscape
print(sprintf('AE_performance_%i_Neighbors_%s.jpg',Neighbors,xscale_chc),'-djpeg','-r300')

%%%Assign score to evaluation data set as score train
eval.features.score_train=score{1};
eval.features.score=score{1};

save(eval_fname,"-struct","eval");