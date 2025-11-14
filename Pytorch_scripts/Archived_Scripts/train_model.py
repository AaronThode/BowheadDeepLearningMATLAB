#train_model.py
# This script processes .wav files to detect bowhead whale calls, extracts spectrogram samples around
# each detection, trains a convolutional autoencoder on these samples, and uses Gaussian Mixture
# Modeling (GMM) to cluster the latent representations learned by the autoencoder.


# %pip install librosa
# %pip install torch torchvision
# %pip install ipykernel

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import random
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

#loaddir = '/Volumes/Bowhead/Shell2010_GSI_Data/S510gsif/S510G0_WAV/'
loaddir= "./"
savedir='OutputDir.dir'
#savedir = '/Users/oceaneboulais/Github/ThodeLab/BowheadWhale/BowheadResults/'
files = [f for f in sorted(os.listdir(loaddir)) if f.lower().endswith('.wav') and not f.startswith('._')]


NFFT = 256  # number of points in each FFT according to Thode et al paper. This corresponds to segment durations of 0.256 seconds at 1 kHz sampling.
specgram_window = plt.mlab.window_hanning(np.ones(NFFT))
# noverlap=28
noverlap = 128  # number of points to overlap between segments according to Thode et al paper. This corresponds to 50% overlap at 1 kHz sampling.
f_hp=30 # high-pass for eliminating low-frequency noise
nu = 1.7 #power law detector
window_sec = 3
pks_idx = []
T = []

fmin = 20
fmax = 475
dB_threshold = 10  # threshold above mean for detection



def calculate_background_median(Pxx, T, window_sec):
    fs_spec = np.round(np.mean(1/np.diff(T)))
    window = int(window_sec*fs_spec)
    xx = median_filter(Pxx, size=(1, window), mode='reflect')
    return xx



for Ifile in range(len(files)):
    y, fs = lb.load(loaddir + files[Ifile], sr=None) # load .wav file
    duration = len(y)/fs  # duration of the audio file in seconds
    # chunk_duration = 60  # duration of each chunk in seconds
    chunk_duration = 3  # duration of each chunk changed to account for predicitve autoencoder
    num_chunks = int(np.ceil(duration/chunk_duration)) # divide the audio file into chunks
    chunk_size = int(chunk_duration*fs)
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx*chunk_size
        end_idx = (chunk_idx + 1)*chunk_size
        if end_idx>len(y):
            end_idx = len(y)
        chunk_y = y[start_idx:end_idx]
        #sos = signal.butter(4, f_hp, 'high', analog=False, output='sos', fs=fs)
       # chunk_y = signal.sosfilt(sos, chunk_y)  # run data through high pass filter
        F, T, Pxx = signal.spectrogram(chunk_y, fs, nperseg=NFFT, nfft=NFFT, noverlap=noverlap,mode='psd', window=specgram_window)  # make spectrogram
        # Pxx = Pxx[3:15, :]  # specify 150-750 Hz data
        # Find freq range indices for 40 to 750 Hz
        
        freq_idx = np.where((F >= fmin) & (F <= fmax))[0]
        Pxx = Pxx[freq_idx, :]

        background = calculate_background_median(Pxx, T, window_sec)
        plstat = np.mean((Pxx/background)**nu, axis=0)  # power law statistic
        #plstat_gauss = 10**(gaussian_filter1d(10*np.log10(plstat), sigma=10)/10)
        # pks_idx, _ = signal.find_peaks(10*np.log10(plstat_gauss), height=10*np.log10(np.mean(plstat_gauss)), distance=72)
       
        plstat_gauss = gaussian_filter1d(10*np.log10(plstat), sigma=10)               
       
        # t=10*np.log10(np.mean(plstat_gauss)) + np.std(10*np.log10(plstat_gauss)), distance=72)  # make detections
        pks_idx, _ = signal.find_peaks(plstat_gauss, height=dB_threshold, distance=72)
        pks_idx = pks_idx[(pks_idx>72) & (pks_idx<len(T)-72)]
        for jj in range(len(pks_idx)):
            T_det = T[int(pks_idx[jj])] + chunk_idx*chunk_duration  # add chunk duration to T_det
            PSD_sample = 10*np.log10(Pxx[:, pks_idx[jj]-72:pks_idx[jj] + 72])  # make spectrogram samples for each detection
            PSD_sample = (PSD_sample-np.mean(PSD_sample))/np.std(PSD_sample)
            PSD_sample = PSD_sample-1
            PSD_sample = np.clip(PSD_sample, 0, 1)
            np.save(savedir + files[Ifile][-19:-4] + '_s' + "{:05.2f}".format(T_det) + '.npy',PSD_sample)  # save .npy file centered at each detection
    print('Processed ' + str(Ifile + 1) + ' files out of ' + str(len(files)))
    print(f"Detections in chunk {chunk_idx} of file {files[Ifile]}: {len(pks_idx)}")
    # plt.plot(10*np.log10(plstat_gauss))
    # plt.scatter(pks_idx, 10*np.log10(plstat_gauss)[pks_idx], color='r')
    # plt.title(f"Detection Statistic with Peaks: {files[ii]}, chunk {chunk_idx}")
    # plt.xlabel("Time index")
    # plt.ylabel("Detection statistic (dB)")

    # plt.show()


    folder_path = savedir # Define the folder containing the detections

batch_size = 64
learning_rate = 0.0001
validation_split = 0.1

#define dataloader for loading detections
class CustomDatasetFull(Dataset):
    def __init__(self, folder_path, transform=None, shuffle=False):
        self.file_list = sorted(os.listdir(folder_path))
        if shuffle:
            random.shuffle(self.file_list)
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        data = np.load(file_path)
        if self.transform:
            data = self.transform(data)
        return data

custom_transform = transforms.ToTensor()

dataset = CustomDatasetFull(folder_path, transform=custom_transform,shuffle=False)

num_samples = len(dataset)
num_train_samples = int((1-validation_split)*num_samples)
num_val_samples = num_samples-num_train_samples
train_dataset, val_dataset = random_split(dataset,[num_train_samples,num_val_samples]) 

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=16,pin_memory=True) #all data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #divide into training and test data
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)

#define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Input spectrograms are 12x144 (so you were close)
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) #stride default is 1, kernal is 3, padding is 1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(4, 1, [3,2], stride=[3,2]) # 16 channels with 18 elements
         # Calculate the size after convolutions and pooling
        self.fc1 = nn.Linear(288, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 288)
        self.pool = nn.MaxPool2d(2, 2)  #kernal size 2, stride is 2 to reduce image dimension size by half, no padding
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))        
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 288)  #flatten the tensor including channels, the size -1 inferred from other dimensions
        latent = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(latent)) #Back to flattend dimensions
        x = x.view(-1, 16, 1, 18) #Reshape back to image16 channels with 18 elements
        x = torch.nn.functional.relu(self.t_conv1(x))
        x = torch.nn.functional.relu(self.t_conv2(x))
        output = torch.sigmoid(self.t_conv3(x))
        return output, latent

latent_dim = 16
device = torch.device("cuda")
autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
autoencoder = autoencoder.float()
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
autoencoder.to(device)

#check connection to GPU
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
   
   Losses = []
ValLosses = []

#set number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    #initialize variables to track total loss
    train_loss_total = 0.0
    val_loss_total = 0.0
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)

    #train autoencoder
    autoencoder.train()
    for data in train_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs, latent = autoencoder(data.float())  
        loss = criterion(outputs, data.float())
        loss.backward()
        optimizer.step()

    #calculate training loss
    autoencoder.eval()
    with torch.no_grad():
        for data in train_dataloader:
            data = data.to(device)
            train_outputs, _ = autoencoder(data.float())  
            train_loss = criterion(train_outputs, data.float())
            train_loss_total += train_loss.item()

    #calculate validation loss
    with torch.no_grad():
        for val_data in val_dataloader:
            val_data = val_data.to(device)
            val_outputs, _ = autoencoder(val_data.float())  
            val_loss = criterion(val_outputs, val_data.float())
            val_loss_total += val_loss.item()

    train_loss_avg = train_loss_total/num_train_batches
    val_loss_avg = val_loss_total/num_val_batches
    Losses.append(train_loss_avg)
    ValLosses.append(val_loss_avg)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}')

torch.save(autoencoder.state_dict(), 'model.pth')

plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
plt.plot(Losses)
plt.plot(ValLosses)
plt.legend(['Training Loss','Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()



#Input all detections into the encoder to extract the latent embeddings
total_len = len(dataloader.dataset)
latent = torch.zeros((total_len, latent_dim), device=device)
with torch.no_grad():
    start_idx = 0
    for i_batch, data in enumerate(dataloader):
        _ , latent_tmp = autoencoder(data.float().to(device))
        end_idx = start_idx + len(latent_tmp)
        latent[start_idx:end_idx] = latent_tmp
        start_idx = end_idx
        print(f"Processed batch {i_batch+1}/{len(dataloader)}")
latent = latent.cpu().numpy()

#use Gaussian mixture modeling to automatically cluster the data using the latent representations
from sklearn.mixture import GaussianMixture
n_clusters = 10 #specify number of clusters
gmm_model = GaussianMixture(n_components=n_clusters,verbose=0,init_params='k-means++',n_init=10)
gmm_result = gmm_model.fit(latent)
clusters = gmm_result.predict(latent)
probabilities = gmm_result.predict_proba(latent)

#Plot sample detections in each GMM cluster
for cc in range(n_clusters):
    print("Samples in Cluster",cc,":")
    cluster_idx = np.where(probabilities[:,cc]>0.99)[0]
    np.random.shuffle(cluster_idx)
    files_plot = dataset.file_list
    plt.figure(0)
    imax=5
    if len(cluster_idx)>=imax:
        for Ifile in range(imax):
            fig = plt.figure(figsize=(5,0.5))
            data_plot = np.load(dataset.folder_path+dataset.file_list[cluster_idx[Ifile]])       
            plt.imshow(data_plot,vmin=0, vmax=1, origin='lower', cmap='inferno',extent=[0,0.36,150,750],aspect='auto')
            plt.gca().invert_yaxis()
            plt.ylabel('freq. (Hz)', fontsize=9)
            plt.xlabel('times (s)', fontsize=9)
            plt.tick_params(axis='y',which='both',right=False,labelright=False,left=True,labelleft=True)
            plt.show()
