function [net_autoencoder, net_decoder,net_dir]=load_trained_network(keyword)

addpath ..
[Database_dir,procdata_basedir,gitpath] = setUpDatabasePaths;

mydir=pwd;
switch keyword
    case 'Original_Manual_NotCentered'
        net_dir=[pwd filesep 'Manual_32latent_NoCentering.dir'];
        Nlatent=32;
        params_file_name='params_2026_03_19__11_02_36.mat';
    case 'Evaluation_Detection_Centered'
        net_dir = [pwd filesep 'Evaluation_Detection_Centered.dir'];
        Nlatent = 32;
        params_file_name = 'params_2026_05_31__11_47_42.mat';

end
addpath(net_dir);
cd(net_dir)
params= load([net_dir filesep params_file_name]);

net_autoencoder = dlnetwork;
net_decoder =dlnetwork;

switch keyword
    case 'Original_Manual_NotCentered'
        tempNet = [
            imageInputLayer([121 104 1],"Name","imageinput_1","Normalization","rescale-zero-one")
            networkLayer(params.TopLevelModule_encoder.Network,"Name",'TopLevelModule:encoder',"OutputNames",{'11'})
            networkLayer(params.TopLevelModule_flatten.Network,"Name",'TopLevelModule:flatten',"OutputNames",{'Flatten0'})
            networkLayer(params.TopLevelModule_to_latent.Network,"Name",'TopLevelModule:to_latent',"OutputNames",{'2'})
            networkLayer(params.TopLevelModule_from_latent.Network,"Name",'TopLevelModule:from_latent',"OutputNames",{'3'})
            networkLayer(params.TopLevelModule_reshape.Network,"Name",'TopLevelModule:reshape',"OutputNames",{'ATEN2'})
            networkLayer(params.TopLevelModule_decoder.Network,"Name",'TopLevelModule:decoder',"OutputNames",{'6'})];

    case 'Evaluation_Detection_Centered'
        %tempNet = [
         %   imageInputLayer([121 104 1],"Name","imageinput_1","Normalization","none")
         %   networkLayer(params.TopLevelModule_encoder.Network,"Name",'TopLevelModule:encoder',"OutputNames",{'11'})
          %  params.TopLevelModule_ATEN5.Layer
         %   networkLayer(params.TopLevelModule_to_latent.Network,"Name",'TopLevelModule:to_latent',"OutputNames",{'2'})];

         tempNet = [
             inputLayer([1 1 120 104],"UUUU","Name","input_1")
    params.TopLevelModule.Layer];
end


net_autoencoder = addLayers(net_autoencoder,tempNet);

% clean up helper variable
clear tempNet;

%%%Finish by initializing the network
net_autoencoder = initialize(net_autoencoder);

net_decoder=[];
switch keyword
    case 'Original_Manual_NotCentered'
        %%%Create 32-element decoder network
        tempNet = [
            inputLayer([Nlatent 1 1 1],"SSCB","Name","input") %The 'C' is channel and 'B' stands for batch and is needed for bulk processing
            %networkLayer(params.TopLevelModule_encoder.Network,"Name",'TopLevelModule:encoder',"OutputNames",{'11'})
            %networkLayer(params.TopLevelModule_flatten.Network,"Name",'TopLevelModule:flatten',"OutputNames",{'Flatten0'})
            %networkLayer(params.TopLevelModule_to_latent.Network,"Name",'TopLevelModule:to_latent',"OutputNames",{'2'})
            networkLayer(params.TopLevelModule_from_latent.Network,"Name",'TopLevelModule:from_latent',"OutputNames",{'3'})
            networkLayer(params.TopLevelModule_reshape.Network,"Name",'TopLevelModule:reshape',"OutputNames",{'ATEN2'})
            networkLayer(params.TopLevelModule_decoder.Network,"Name",'TopLevelModule:decoder',"OutputNames",{'6'})];
        net_decoder = addLayers(net_decoder,tempNet);

        % clean up helper variable
        clear tempNet;

        net_decoder = initialize(net_decoder);

end

cd(mydir)