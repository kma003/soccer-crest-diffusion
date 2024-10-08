import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset
from params.defs import DATA_DIR

class SoccerCrestsDataset(Dataset):
    def __init__(self, dataset_name, transform=None):
        # Read in data from csv
        csv_file = os.path.join(DATA_DIR,dataset_name,'training_data.csv')
        self.training_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.training_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        country_name = self.training_df.iloc[idx,0]
        team_name = self.training_df.iloc[idx,1]
        #img = io.imread(self.training_df.iloc[idx,2])
        img = plt.imread(self.training_df.iloc[idx,2])

        # Only keep the RGB dimension if image is a PNG, and turn the background white
        if img.shape[-1] == 4:
            alpha_channel = img[:,:,3]
            img = img[:,:,:3]
            mask = alpha_channel == 0
            img[mask] = [1.0,1.0,1.0] #img[mask] = 255

        # TODO This could happen during Transforms
        # TODO Just switch to opencv
        # Resize to 256 x 256 and convert to float32
        img = transform.resize(img, output_shape=(256,256)).astype(np.float32) # TODO Don't hardcode this image size
        #img = img / 255.0 # Normalize to be between 0 and 1
        img = np.clip(img, 0, 1) # Make sure all values are between 0 and 1
        img = img * 2 - 1 # Do final scaling to be between -1 and 1

        if self.transform:
            img = self.transform(img)

        sample = {'image':img,'team':team_name,'country':country_name}

        return sample
