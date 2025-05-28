import pickle
import numpy as np
import os
import torch
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    ResizeWithPadOrCropd,
)

MODALITY_MAP = {"t1": 0.0, "t2": 1.0}
STUDY_MAP = {"hcpya": 'hcpya_relpaths_and_metadata.pkl', "hcpdev": 'hcpdev_relpaths_and_metadata.pkl', "hcpag": 'hcpag_relpaths_and_metadata.pkl', "openneuro": 'openneuro_relpaths_and_metadata.pkl', "abcd": 'abcd_relpaths_and_metadata.pkl'}


def perform_data_qc(l):
    """Throws out data if sex is not 0/1 or if age is NaN (or invalid numeric).
    Ignores 'OTHER' for 's' (sex)."""

    def safe_convert_to_float(x):
        """
        Attempts to convert x to float.
        If x is a string, we strip whitespace first.
        Raise ValueError if it's 'OTHER' or otherwise invalid.
        """
        # If it's a string, strip whitespace
        if isinstance(x, str):
            x = x.strip()
            # Handle special placeholder
            if x == "OTHER":
                raise ValueError("Encountered 'OTHER'.")

        return float(x)  # Will raise ValueError/TypeError if not convertible

    qc = []
    for d in l:
        p = d["path"]
        a = d["age"]
        s = d["sex"]
        m = d["modality"]
        try:
            a_val = safe_convert_to_float(a)
            s_val = safe_convert_to_float(s)
            m_val = safe_convert_to_float(m)
        except (ValueError, TypeError):
            # If a, s, or m are "OTHER" or can't be converted
            print(f"Skipping invalid row: (p={p}, a={a}, s={s}, m={m})")
            continue

        # Check for NaN
        if math.isnan(a_val) or math.isnan(s_val):
            print(f"Found NaN in row: (p={p}, a={a}, s={s}, m={m})")
            continue

        # If all checks pass, add it
        qc.append({"path": p, "age": a_val, "sex": s_val, "modality": m_val})

    return qc


def get_file_list_bwm(studies=["hcpya", "hcpdev", "hcpag", "openneuro", "abcd"], modality=("t1", "t2"), verbose=True, unpaired_modality=True):
    """Returns file list for data in BWM directory."""

    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        root_dir = "/simurgh/group/BWM/"
        pkl_files = [os.path.join(root_dir, "pkl", STUDY_MAP[study]) for study in studies]
    elif cluster_name == "haic":
        root_dir = "/hai/scratch/alanqw/BWM/"
        pkl_files = [os.path.join(root_dir, "pkl", STUDY_MAP[study]) for study in studies]
    elif cluster_name == "sherlock":
        raise NotImplementedError
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )

    if isinstance(modality, str):
        modality = [modality]

    train_data = []
    val_data = []

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Prepend root file path
        for d in data:
            if "t1_path" in d:
                d["t1_path"] = os.path.join(root_dir, d["t1_path"])
            if "t2_path" in d:
                d["t2_path"] = os.path.join(root_dir, d["t2_path"])

        # Unpaired modality
        if unpaired_modality:
            unpaired_data = []
            for d in data:
                if "t1_path" in d:
                    unpaired_data.append(
                        {
                            "path": d["t1_path"],
                            "age": d["age"],
                            "sex": d["sex"],
                            "modality": MODALITY_MAP["t1"],
                        }
                    )
                if "t2_path" in d:
                    unpaired_data.append(
                        {
                            "path": d["t2_path"],
                            "age": d["age"],
                            "sex": d["sex"],
                            "modality": MODALITY_MAP["t2"],
                        }
                    )
            data = unpaired_data

        # Split data into train and val
        train_split, val_split = (
            data[: int(len(data) * 0.8)],
            data[int(len(data) * 0.8) :],
        )
        train_data.extend(train_split)
        val_data.extend(val_split)

    if verbose:
        print("\nTotal number of train paths before QC: ", len(train_data))
        print("Total number of val paths before QC: ", len(val_data))
        print("Total number of images before QC: ", len(train_data) + len(val_data))

    train_data = perform_data_qc(train_data)
    val_data = perform_data_qc(val_data)

    if verbose:
        print("\nTotal number of train paths after QC: ", len(train_data))
        print("Total number of val paths after QC: ", len(val_data))
        print("Total number of images after QC: ", len(train_data) + len(val_data))

    return train_data, val_data


class FileListDataset(Dataset):
    def __init__(
        self,
        file_list,
        condition_list=None,
        transform=None,
        data_key="image",
    ):
        self.file_list = file_list
        self.transform = transform
        self.condition_list = condition_list
        self.data_key = data_key

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = {self.data_key: img_path}

        if self.transform:
            data = self.transform(data)

        condition_tensor = self.condition_list[idx]
        data["img_path"] = img_path
        data["age"] = condition_tensor[0]
        data["sex"] = condition_tensor[1]
        data["modality"] = condition_tensor[2]
        return data


class BWM:
    def __init__(
        self,
        img_size,
        num_workers,
        read_from_scr=True,
        modality=("t1", "t2"),
        age_normalization=None,
        rank=0,
        world_size=1,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=False,
        studies=["hcpya", "hcpdev", "hcpag", "openneuro", "abcd"],
    ):
        self.num_workers = num_workers
        assert age_normalization in [
            None,
            "zscore",
            "min-max",
        ], "Choose a valid age normalization method"
        self.age_normalization = age_normalization
        self.modality = modality
        self.rank = rank
        self.world_size = world_size
        self.ddp_bool = world_size > 1
        self.data_key = data_key
        self.sample_balanced_age_for_training = sample_balanced_age_for_training
        self.read_from_scr = read_from_scr
        self.studies = studies

        self.age_mu = 0
        self.age_sigma = 1

        self.train_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key], channel_dim="no_channel"),
                EnsureTyped(keys=[data_key], dtype=torch.float32),
                Orientationd(keys=[data_key], axcodes="RAS"),
                Spacingd(keys=[data_key], pixdim=spacing, mode=("bilinear")),
                ResizeWithPadOrCropd(keys=[data_key], spatial_size=img_size),
                ScaleIntensityRangePercentilesd(
                    keys=data_key, lower=0, upper=99.5, b_min=0, b_max=1, clip=True
                ),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key], channel_dim="no_channel"),
                EnsureTyped(keys=[data_key], dtype=torch.float32),
                Orientationd(keys=[data_key], axcodes="RAS"),
                Spacingd(keys=[data_key], pixdim=spacing, mode=("bilinear")),
                ResizeWithPadOrCropd(keys=[data_key], spatial_size=img_size),
                ScaleIntensityRangePercentilesd(
                    keys=data_key, lower=0, upper=99.5, b_min=0, b_max=1, clip=True
                ),
            ]
        )

    @staticmethod
    def _zscore_normalize(x, mu, sigma):
        return (x - mu) / (sigma + 1e-8)

    @staticmethod
    def _inverse_zscore_normalize(x, mu, sigma):
        return x * sigma + mu

    @staticmethod
    def _min_max_scale(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-8)

    @staticmethod
    def _inverse_min_max_scale(x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def normalize_age(self, ages):
        if self.age_normalization == "zscore":
            ages = self._zscore_normalize(ages, self.age_mu, self.age_sigma)
        elif self.age_normalization == "min-max":
            ages = self._min_max_scale(ages, self.age_min, self.age_max)
        return ages

    def get_dataloaders(self, batch_size, drop_last=False):
        train_data, val_data = get_file_list_bwm(studies=self.studies, modality=self.modality)
        train_paths = [d["path"] for d in train_data]
        val_paths = [d["path"] for d in val_data]
        train_ages = [d["age"] for d in train_data]
        val_ages = [d["age"] for d in val_data]
        train_sexes = [d["sex"] for d in train_data]
        val_sexes = [d["sex"] for d in val_data]
        train_modalities = [d["modality"] for d in train_data]
        val_modalities = [d["modality"] for d in val_data]

        train_ages = np.array(train_ages)
        val_ages = np.array(val_ages)
        if self.age_normalization == "zscore":
            # Z-score normalization for age
            self.age_mu = train_ages.mean()
            self.age_sigma = train_ages.std()
            train_ages = self._zscore_normalize(train_ages, self.age_mu, self.age_sigma)
            val_ages = self._zscore_normalize(val_ages, self.age_mu, self.age_sigma)
        elif self.age_normalization == "min-max":
            self.age_min = train_ages.min()
            self.age_max = train_ages.max()
            train_ages = self._min_max_scale(train_ages, self.age_min, self.age_max)
            val_ages = self._min_max_scale(val_ages, self.age_min, self.age_max)

        # Zip the conditions into one single list
        train_conditions = [
            (a, b, c) for a, b, c in zip(train_ages, train_sexes, train_modalities)
        ]
        val_conditions = [
            (a, b, c) for a, b, c in zip(val_ages, val_sexes, val_modalities)
        ]

        train_ds = FileListDataset(
            train_paths,
            condition_list=train_conditions,
            transform=self.train_transforms,
            data_key=self.data_key,
        )

        val_ds = FileListDataset(
            val_paths,
            condition_list=val_conditions,
            transform=self.val_transforms,
            data_key=self.data_key,
        )

        if self.ddp_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, num_replicas=self.world_size, rank=self.rank
            )
            shuffle = False
        elif self.sample_balanced_age_for_training:
            train_sampler = self.get_age_balanced_sampler(train_ages)
            val_sampler = None
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        print("shuffle for train: ", (not self.ddp_bool))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=train_sampler,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=val_sampler,
            drop_last=drop_last,
        )
        return train_loader, val_loader

    def get_age_balanced_sampler(self, ages, num_bins=10):
        """
        Create a weighted sampler that balances samples across age bins,
        agnostic to the units of the age values (e.g., years, normalized, etc.).

        Parameters:
            ages (iterable): A list or array of age values.
            num_bins (int): The number of bins to use for balancing. Default is 10.

        Returns:
            WeightedRandomSampler: A sampler that samples items inversely
                                proportional to the frequency of their age bin.
        """
        ages = np.array(ages)
        print("Input ages:", ages)

        # Handle the edge case where all ages are nearly identical.
        if np.allclose(ages.min(), ages.max()):
            print(
                "All ages are approximately equal. Assigning equal weights to all samples."
            )
            sample_weights = np.ones_like(ages, dtype=float)
        else:
            print(f"Age range: min = {ages.min()}, max = {ages.max()}")
            # Create bins that span the range of the age values.
            bins = np.linspace(ages.min(), ages.max(), num_bins + 1)
            print("Computed bins:", bins)

            # Determine bin indices for each age.
            # Adjusting the index so that each age falls into a bin between 0 and (num_bins - 1)
            bin_indices = np.searchsorted(bins, ages, side="right") - 1
            print("Assigned bin indices:", bin_indices)

            # Compute the count of samples in each bin.
            unique_bins, counts = np.unique(bin_indices, return_counts=True)
            print("Unique bins and their counts:")
            for bin_val, count in zip(unique_bins, counts):
                print(f"  Bin {bin_val}: {count} sample(s)")

            # Compute weights for each bin as the inverse of the count.
            bin_weights = {
                bin_idx: 1.0 / count for bin_idx, count in zip(unique_bins, counts)
            }
            print("Calculated bin weights (inverse frequency):", bin_weights)

            # Map each sample to the weight corresponding to its bin.
            sample_weights = [bin_weights[idx] for idx in bin_indices]
            # print("Final sample weights:", sample_weights)

        # Create the weighted random sampler.
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        print(f"Created WeightedRandomSampler with {len(sample_weights)} samples.")

        return sampler


if __name__ == "__main__":
    dataset = BWM(
        img_size=(128, 128, 128),
        num_workers=4,
        zscore_age=True,
        rank=0,
        world_size=1,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=False,
    )

    import matplotlib.pyplot as plt

    train_loader, val_loader = dataset.get_dataloaders(batch_size=4)
    for data in train_loader:
        print(data)
        plt.imshow(data["image"][0, 0, 80, :, :])
        plt.show()
        break
    for data in val_loader:
        print(data)
        plt.imshow(data["image"][0, 0, 80, :, :])
        plt.show()
        break
