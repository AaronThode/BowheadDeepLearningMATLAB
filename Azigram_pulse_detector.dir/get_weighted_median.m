function [median_compass,iqr_out]=get_weighted_median(mask,azi,B,da)
%function [median_compass,iqr_out]=get_weighted_median(mask,azi,B,da)

%%%No longer weighted median, but circular median
azi_samp=mask.*azi;
P_samp=mask.*B;
azi_samp=azi_samp(:);P_samp=P_samp(:);
Igood=(mask==1);
azi_samp=azi_samp(Igood);P_samp=P_samp(Igood);



%Convert compass angles (defined clockwise relative to y-axis) into standard math angles
%  (\theta \in (-pi,pi] )

azi_math=90 - azi_samp;   %%%Angles now defined c-clockwise relative to x axis
Ifix=(azi_math<-180);
azi_math(Ifix)=azi_math(Ifix)+360;

% compute median and std
median_math=180*circ_median(azi_math*pi/180)/pi;  %Original existing line
iqr_out=180*circ_std(azi_math*pi/180)/pi;

%Convert back into compass convention.
median_compass=90-median_math;
median_compass(median_compass<0)=median_compass(median_compass<0)+360;


%Convert back into compass convention.
if median_compass<0
    keyboard
    median_compass=median_compass+360;
end

% adjust iqr
if iqr_out<0
    keyboard
    iqr_out = iqr_out+360;
end


%%%%Debug figures
if 1==0
    figure(1000); subplot(2,1,1);histogram(azi_samp,0:2:360);
    hold on;plot(median_compass,0,'o','markersize',10);hold off;
    subplot(2,1,2); histogram(azi_math,-180:2:180);
    hold on;plot(median_math,0,'o','markersize',10);hold off;
    
end

%%%figure;histogram(azi_samp,0:2:360)
%%%% grid on;
%%%% xlabel('Azimuth');ylabel('Count');set(gca,'fontweight','bold','fontsize',14);
%%%The approach below caused odd results (gaps) in the localization,
%%% esp. for DASAR Y at Eel cove: fixed 11/23/2019.

%%%
%%%If we're computing bearings, check that
%%% there is not a discontinous jump
% transform_flag=false;
% if (max(azi_samp)-min(azi_samp)>2*da)  %%%We have more of a spread in bearings than is possible
%    azi_samp=sind(azi_samp);
%    transform_flag=true;
% end

% [output,iqr_out]=weightedMedian(azi_samp,P_samp);
%
% if transform_flag
%    output=asind(output);
%    iqr_out=asind(iqr_out);
%    if output<0
%       output=360+output;
%    end
% end