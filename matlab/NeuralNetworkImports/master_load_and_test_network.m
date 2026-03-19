%%%%%master_load_and_test_network.m%%%%%%

network_chc='Original_Manual_NotCentered';

[net_autoencoder,net_decoder,net_dir]=load_trained_network(network_chc);

%input_data=load('S314D0T20140901T131043_Type2.mat');
input_data=load('S508G0T20080828T163252_Type1.mat');

input_image=single(input_data.SNR_gram)./single(max(max(input_data.SNR_gram)));
output_image=predict(net_autoencoder,input_image);
latent=predict(net_autoencoder,input_image,'Outputs','TopLevelModule:to_latent');
output_image_decoder=predict(net_decoder,latent);

figure(100)
subplot(3,1,1)
imagesc(input_data.SNR_gram);axis xy


subplot(3,1,2)
imagesc(output_image);title('Autoencoder output');
grid on;axis xy

subplot(3,1,3)
imagesc(output_image_decoder);title('Decoder output');
grid on;axis xy

%%%IMPORTANT!
%  Remove network from path in case we want to run different network in
%%%future

rmpath(net_dir);