%%%master_summarize_edits_to_latent_files
%  Assumes the latent file has been loaded into 'data'.

reviewer='XX';

Ntotal=length(data.reviewer);

data.features.type(data.features.type==10)=-10;
data.features.type(data.features.type==12)=-3;

Ncall_org=sum(data.features.type_org>0);
Nother_org=Ntotal-Ncall_org;
fprintf('In original dataset are %i calls and %i other...\n',Ncall_org,Nother_org);

Ncall=sum(data.features.type>0 );
Nother=Ntotal-Ncall;
fprintf('In revised dataset are %i calls and %i other...\n',Ncall,Nother);



fprintf('%s has reviewed %i out ot %i total samples.\n',reviewer,sum(contains(data.reviewer,reviewer)),length(data.reviewer));

Nchanged=sum(data.features.type~=data.features.type_org);
fprintf('%i Samples have been changed.\n',Nchanged)


%%%What whale calls have been reclassified as other?
Category=[0 1 2 3 10];
Cat_name={'No signal','Multiple Signals','Airgun+Call','Uncertain','Man-made (airgun or vessel}'};
%Ncat=[Nother Nother Nother Nother Ncall Nother];

for I=1:length(Category)
    Nalt(I)=sum(data.features.type==-Category(I) & data.features.type_org>0);
    fprintf('%i samples (%3.2f percent) have been changed from CD to %s...\n',Nalt(I),100*Nalt(I)/Ncall_org,Cat_name{I});
end

N_CD_to_OT=sum(data.features.type<=0& data.features.type_org>0);
fprintf('%i samples (%3.2f percent) changed from whale calls to transients\n',N_CD_to_OT,100*N_CD_to_OT/Ncall_org);

%%%What transients have been reclassified as whale calls?

N_OT_to_CD=sum(data.features.type_org<=0  & data.features.type>0);
fprintf('%i samples (%3.2f percent) changed from transients to whale calls\n',N_OT_to_CD,100*N_OT_to_CD/Nother_org);