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

plt.style.use('_mpl-gallery-nogrid')

#loaddir = '/Volumes/Bowhead/Shell2010_GSI_Data/S510gsif/S510G0_WAV/'
loaddir= "./"
savedir='/Users/thode/Desktop/BowheadEvents.dir/'

my_debug=False
dB_threshold = 20  # threshold above mean for detection
image_scale_factor = 5  # factor to multiply SNR by for saving as unit8 image  
fmin = 10
fmax = 475
window_sample_sec = 3 # seconds to take for each sample, centered on peak SNR

#savedir = '/Users/oceaneboulais/Github/ThodeLab/BowheadWhale/BowheadResults/'
files = [f for f in sorted(os.listdir(loaddir)) if f.lower().endswith('.wav') and not f.startswith('._')]


NFFT = 256  # number of points in each FFT according to Thode et al paper. This corresponds to segment durations of 0.256 seconds at 1 kHz sampling.
specgram_window = plt.mlab.window_hanning(np.ones(NFFT))
# noverlap=28
noverlap = 128+64  # number of points to overlap between segments according to Thode et al paper. This corresponds to 50% overlap at 1 kHz sampling.
#f_hp=30 # high-pass for eliminating low-frequency noise
nu = 1.7 #power law detector
window_sec_median = 5  # median filter window in seconds
chunk_duration = 60  # duration of each chunk for processing detections
   
#min_distance =4  # minimum distance between detections in samples (at 4 Hz this is 0.75 seconds)

pks_idx = []
T = []



def calculate_background_median(Pxx, T, window_sec):
    fs_spec = np.round(np.mean(1/np.diff(T)))
    window = int(window_sec*fs_spec)
    xx = median_filter(Pxx, size=(1, window), mode='reflect')
    return xx



for Ifile in range(len(files)):
    counts = 0
    y, fs = lb.load(loaddir + files[Ifile], sr=None) # load .wav file
    duration = len(y)/fs  # duration of the audio file in seconds

    Iwindow_sample=int(np.ceil(0.5*fs*window_sample_sec/(NFFT-noverlap)))
    # chunk_duration = 60  # duration of each chunk in seconds
    num_chunks = int(np.ceil(duration/chunk_duration)) # divide the audio file into chunks
    chunk_size = int(chunk_duration*fs)
    for Ichunk in range(num_chunks):
        start_idx = Ichunk*chunk_size
        end_idx = (Ichunk + 1)*chunk_size
        if end_idx>len(y):
            end_idx = len(y)
        chunk_y = y[start_idx:end_idx]
        #sos = signal.butter(4, f_hp, 'high', analog=False, output='sos', fs=fs)
       # chunk_y = signal.sosfilt(sos, chunk_y)  # run data through high pass filter
        F, T, Pxx = signal.spectrogram(chunk_y, fs, nperseg=NFFT, nfft=NFFT, noverlap=noverlap,mode='psd', window=specgram_window)  # make spectrogram
        # Pxx = Pxx[3:15, :]  # specify 150-750 Hz data
        # Find freq range indices for 40 to 750 Hz
        
        Ifreq = np.where((F >= fmin) & (F <= fmax))[0]
        Pxx = Pxx[Ifreq, :]

        
        background = calculate_background_median(Pxx, T, window_sec_median)
        plstat = 10*np.log10(np.mean((Pxx/background)**nu, axis=0))  # power law statistic
        #plstat_gauss = 10**(gaussian_filter1d(10*np.log10(plstat), sigma=10)/10)
        # pks_idx, _ = signal.find_peaks(10*np.log10(plstat_gauss), height=10*np.log10(np.mean(plstat_gauss)), distance=72)
       

        #plstat_gauss = gaussian_filter1d(plstat, sigma=2)               
       
        # t=10*np.log10(np.mean(plstat_gauss)) + np.std(10*np.log10(plstat_gauss)), distance=72)  # make detections
        pks_idx, _ = signal.find_peaks(plstat, height=dB_threshold, distance=Iwindow_sample)
        #results_full = signal.peak_widths(plstat, pks_idx, rel_height=0.9)

        if my_debug:
            fig, ax = plt.subplots(layout='constrained')
            ax.imshow(10*np.log10(Pxx), origin='lower')

       # ax.pcolormesh(10*np.log10(Pxx))
       # plt.plot(T,plstat)
            fig, ax = plt.subplots(layout='constrained')
            plt.plot(plstat)
        #plt.hlines(*results_full[1:], color="C2")
            ax.scatter(pks_idx,plstat[pks_idx])
            ax.grid(True)
            plt.show()
        pks_idx = pks_idx[(pks_idx>Iwindow_sample) & (pks_idx<len(T)-Iwindow_sample)]

       
        for Ipeak in range(len(pks_idx)):
            T_det = T[int(pks_idx[Ipeak])] + Ichunk*chunk_duration  # add chunk duration to T_det
            
            PSD_sample = 10*np.log10(Pxx[:, (pks_idx[Ipeak]-Iwindow_sample):(pks_idx[Ipeak] + Iwindow_sample)])  # make spectrogram samples for each detection
            
               
    # matrix = np.array([[10, 20, 30],
    #                    [40, 50, 60],
    #                    [70, 80, 90]])
    # vector = np.array([1, 2, 3])

    # result = matrix/vector[:, np.newaxis]
    # print(result)

            median_sample = np.median(PSD_sample, axis=1)
            #std_sample = np.std(PSD_sample, axis=1)
            SNR_sample = image_scale_factor*(PSD_sample-median_sample[:,np.newaxis])
            # multiplying by 10 gives a SNR resolution of 0.1 dB
            SNR_sample[SNR_sample < 0] = 0
            SNR_sample[SNR_sample > 255] = 255
            SNR_sample8 = SNR_sample.astype('uint8')
           
            if my_debug:
                fig = plt.figure(figsize=(15, 9)) #width, height in inches
                ax0 = fig.add_subplot(1, 3, 1)
                im0=plt.imshow(PSD_sample, cmap='gray', origin='lower')
                ax0.set_title('Power Spectral Density (dB)')
                fig.colorbar(im0, ax=ax0)

                ax1 = fig.add_subplot(1, 3, 2)
                #fig, ax = plt.subplots(layout='constrained')
                im=plt.imshow(SNR_sample/image_scale_factor, cmap='gray', origin='lower')
                ax1.set_title('float image with median removed at each frequency band')
                fig.colorbar(im, ax=ax1)
                print('float image with median removed at each frequency band')

                ax2 = fig.add_subplot(1, 3, 3)
                im2=plt.imshow(SNR_sample8, cmap='gray', origin='lower')
                ax2.set_title('unit8 image, SNR multiplied by image_scale_factor')
                fig.colorbar(im2, ax=ax2)
                plt.draw()
                plt.pause(2)  # Pause to ensure the plot updates
                plt.close('all')
               # fig, ax = plt.subplots(layout='constrained')
               # im=ax.imshow(PSD_sample, origin='lower')
                #ax.set_title('image with median removed at each frequency band')
                #fig.colorbar(im, ax=ax)
               # print('image with median removed at each frequency band')

            savestr=savedir + files[Ifile][-19:-4] + '_s' + "{:05.2f}".format(T_det) + '.npy'
            np.save(savestr,SNR_sample8)  # save .npy file centered at each detection
            counts +=1
            if np.remainder(counts, 100)==0:
                print(counts)
    print(f"File {Ifile+1}/{len(files)}, chunk {Ichunk+1}/{num_chunks}, detections {counts}")
    print('Processed ' + str(Ifile + 1) + ' files out of ' + str(len(files)))
    print(f"Detections in file {files[Ifile]}: {counts}")
     
    # plt.scatter(pks_idx, 10*np.log10(plstat_gauss)[pks_idx], color='r')
    # plt.title(f"Detection Statistic with Peaks: {files[ii]}, chunk {chunk_idx}")
    # plt.xlabel("Time index")
    # plt.ylabel("Detection statistic (dB)")

    # plt.show()

