import os
import numpy as np

from monai.data import DataLoader, Dataset


class NPZDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.files = [f for f in os.listdir(directory) if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the .npz file
        file_path = os.path.join(self.directory, self.files[idx])
        data = np.load(file_path)
        return data


def create_dataloader(directory, batch_size=1, shuffle=False):
    dataset = NPZDataset(directory=directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
