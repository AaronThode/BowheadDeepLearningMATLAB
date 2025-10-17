import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
import random
import matplotlib.pyplot as plt


relative_path_from_cwd = os.path.join("data", "input.txt")
print(f"Path relative to CWD: {relative_path_from_cwd}")

    # Relative path navigating up a directory
relative_path_up_one = os.path.join("..", "config", "settings.ini")
print(f"Path navigating up: {relative_path_up_one}")

savedir='/Users/thode/Desktop/BowheadEvents.dir/'
folder_path = savedir # Define the folder containing the detections
image_scale_factor = 5  # factor to multiply SNR by for saving as unit8 image


batch_size = 64
learning_rate = 0.0001
validation_split = 0.2

#define dataloader for loading detections

filelist = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.npy')]
file_path = os.path.join(folder_path, filelist[0])
image = np.load(file_path)
nrow,ncol = image.shape
print("nrow,ncol=",nrow,ncol)


class CustomDatasetFull(Dataset):
    def __init__(self, folder_path, transform=None, shuffle=False):
        self.folder_path = folder_path
        self.file_list = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.npy')]
        if shuffle:
            random.shuffle(self.file_list)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        image = np.load(file_path)
       
        #image = transforms.ToTensor()(image)
       
       # fig, ax = plt.subplots(layout='constrained')
       # ax.imshow(image, origin='lower')
        my_debug = False
        if my_debug:
            print(file_path)
            fig = plt.figure(figsize=(15, 9)) #width, height in inches
            ax0 = fig.add_subplot(1, 2, 1)
            im0=plt.imshow(image, cmap='gray', origin='lower')
            ax0.set_title('Input image')
            fig.colorbar(im0, ax=ax0)


        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
            if image.ndim == 2:  # If grayscale, add channel dimension
                image = image.unsqueeze(0)

        if my_debug:
            ax1 = fig.add_subplot(1, 2, 2)
            im1=plt.imshow(image[0,:,:], cmap='gray', origin='lower')
            ax1.set_title('Converted image')
            fig.colorbar(im1, ax=ax1)
            plt.draw()
            plt.pause(5)  # Pause to ensure the plot updates
            plt.close('all')
               
        return image
        

custom_transform = transforms.ToTensor()

dataset = CustomDatasetFull(folder_path, transform=custom_transform,shuffle=False)
#determine size of input images
#datatemp = next(iter(train_dataloader))
print(dataset[0].size())
image_dims=dataset[0].size()
image_dims=image_dims[1:] #remove channel dimension

num_samples = len(dataset)
num_train_samples = int((1-validation_split)*num_samples)
num_val_samples = num_samples-num_train_samples
train_dataset, val_dataset = random_split(dataset,[num_train_samples,num_val_samples]) 

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=16,pin_memory=True) #all data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #divide into training and test data
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)



#define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, latent_dim): #Defines the structure of the autoencoder
        super(Autoencoder, self).__init__()  #need a sequential step?
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) 
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(4, 1, [3,2], stride=[3,2])
        self.fc1 = nn.Linear(288, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 288)
        self.pool = nn.MaxPool2d(2, 2)  #AdaptiveAvgPool maybe better?
    def forward(self, x): #when running the model, this is the function that is called
        x = torch.nn.functional.relu(self.conv1(x))        
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 288)
        latent = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(latent))
        x = x.view(-1, 16, 1, 18)
        x = torch.nn.functional.relu(self.t_conv1(x))
        x = torch.nn.functional.relu(self.t_conv2(x))
        output = torch.sigmoid(self.t_conv3(x))
        return output, latent




latent_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
autoencoder = autoencoder.float()
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate) #optimizer = optim.Adam(model.parameters(), lr=0.001)
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
