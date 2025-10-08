%  Imatch=evaluate_overlap_between_manual_automated(manual.tsec,manual.tend,detect.tstart,detect.tend,param.compare.ovlap);

function [score,manual_index]=evaluate_overlap_between_manual_automated(t1_s,t1_e,t2_s,t2_e,ovlap)

t1_s=t1_s(:);
t1_e=t1_e(:);
t2_s=t2_s(:);
t2_e=t2_e(:);
score=NaN(length(t2_s),1);
manual_index=NaN(length(t2_s),1);
for I=1:length(t1_s)


    tmin=max([repmat(t1_s(I),length(t2_s),1) t2_s],[],2);
    tmax=min([repmat(t1_e(I),length(t2_e),1) t2_e],[],2);

    duration_1=t1_e-t1_s;
    duration_2=t2_e-t2_s;

    min_duration=min([repmat(duration_1(I),length(duration_2),1) duration_2],[],2);
    [frac_ovlap,Imax]=max((tmax-tmin)./min_duration);
    %score(Imax)=frac_ovlap>=ovlap;
    score(Imax)=frac_ovlap;
    manual_index(Imax)=I;
  %  if Imax==193
  %      keyboard
  %  end


end
keyboard
