import pickle
import numpy as np
import os
import torch
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


def get_t1_all_file_list():
    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        prefix = "/simurgh/u/fangruih"
        file_dir_prefix = "/simurgh/u/alanqw/data/fangruih/stru/"
    elif cluster_name == "haic":
        prefix = "/hai/scratch/fangruih"
        file_dir_prefix = "/hai/scratch/fangruih/data/"
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )

    dataset_names = [
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/abcd/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/adni_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_ag_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_dev_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_ya_mpr1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/ppmi_t1/paths_and_info_flexpath.pkl",
    ]
    train_images = []
    train_ages = []
    train_sexes = []
    val_images = []
    val_ages = []
    val_sexes = []

    for dataset_name in dataset_names:
        with open(dataset_name, "rb") as file:
            data = pickle.load(file)

            # Convert paths and ages to lists if they are NumPy arrays
            train_new_images = (
                data["train"]["paths"].tolist()
                if isinstance(data["train"]["paths"], np.ndarray)
                else data["train"]["paths"]
            )
            train_new_ages = (
                data["train"]["age"].tolist()
                if isinstance(data["train"]["age"], np.ndarray)
                else data["train"]["age"]
            )
            train_new_sex = (
                data["train"]["sex"].tolist()
                if isinstance(data["train"]["sex"], np.ndarray)
                else data["train"]["sex"]
            )

            val_new_images = (
                data["val"]["paths"].tolist()
                if isinstance(data["val"]["paths"], np.ndarray)
                else data["val"]["paths"]
            )
            val_new_ages = (
                data["val"]["age"].tolist()
                if isinstance(data["val"]["age"], np.ndarray)
                else data["val"]["age"]
            )
            val_new_sex = (
                data["val"]["sex"].tolist()
                if isinstance(data["val"]["sex"], np.ndarray)
                else data["val"]["sex"]
            )

            train_images += train_new_images
            train_ages += train_new_ages
            train_sexes += train_new_sex

            val_images += val_new_images
            val_ages += val_new_ages
            val_sexes += val_new_sex

    train_images = [file_dir_prefix + train_image for train_image in train_images]
    val_images = [file_dir_prefix + val_image for val_image in val_images]

    print(len(train_images))
    print(len(val_images))

    return train_images, train_ages, train_sexes, val_images, val_ages, val_sexes


class FileListDataset(Dataset):
    def __init__(
        self,
        file_list,
        transform,
        condition_list=None,
        data_key="image",
    ):
        self.file_list = file_list
        self.transform = transform
        self.condition_list = condition_list
        self.data_key = data_key

        if self.condition_list is not None:
            assert len(self.file_list) == len(
                self.condition_list
            ), "File list and condition list should have the same length."

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = {self.data_key: img_path}

        if self.transform:
            data = self.transform(data)

        if self.condition_list is not None:
            condition_tensor = self.condition_list[idx]
            data["age"] = condition_tensor[0]
            data["sex"] = condition_tensor[1]
        return data


class T1All:
    def __init__(
        self,
        img_size,
        num_workers,
        zscore_age=False,
        rank=0,
        world_size=1,
        channel=0,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=False,
    ):
        self.num_workers = num_workers
        self.zscore_age = zscore_age
        self.rank = rank
        self.world_size = world_size
        self.ddp_bool = world_size > 1
        assert channel in [0, 1, 2, 3], "Choose a valid channel"
        self.data_key = data_key
        self.sample_balanced_age_for_training = sample_balanced_age_for_training

        self.age_mu = 0
        self.age_sigma = 1

        self.train_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key]),
                Lambdad(keys=data_key, func=lambda x: x[channel, :, :, :]),
                EnsureChannelFirstd(keys=[data_key], channel_dim=0),
                EnsureTyped(keys=[data_key]),
                Orientationd(keys=[data_key], axcodes="RAS"),
                Spacingd(keys=[data_key], pixdim=spacing, mode=("bilinear")),
                ResizeWithPadOrCropd(keys=[data_key], spatial_size=img_size),
                # train_crop_transform,
                ScaleIntensityRangePercentilesd(
                    keys=data_key, lower=0, upper=99.5, b_min=0, b_max=1
                ),
                EnsureTyped(keys=data_key, dtype=torch.float32),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key]),
                Lambdad(keys=data_key, func=lambda x: x[channel, :, :, :]),
                EnsureChannelFirstd(keys=[data_key], channel_dim=0),
                EnsureTyped(keys=[data_key]),
                Orientationd(keys=[data_key], axcodes="RAS"),
                Spacingd(keys=[data_key], pixdim=spacing, mode=("bilinear")),
                ResizeWithPadOrCropd(keys=[data_key], spatial_size=img_size),
                # CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
                ScaleIntensityRangePercentilesd(
                    keys=data_key, lower=0, upper=99.5, b_min=0, b_max=1
                ),
                EnsureTyped(keys=data_key, dtype=torch.float32),
            ]
        )

    def zscore_normalize_age(self, age):
        return (age - self.age_mu) / (self.age_sigma + 1e-8)

    def zscore_unnormalize_age(self, age):
        return age * self.age_sigma + self.age_mu

    def get_dataloaders(
        self,
        batch_size,
        drop_last=False,
    ):
        train_images, train_ages, train_sexes, val_images, val_ages, val_sexes = (
            get_t1_all_file_list()
        )

        train_ages = np.array(train_ages)
        val_ages = np.array(val_ages)
        if self.zscore_age:
            # Z-score normalization for age
            self.age_mu = train_ages.mean()
            self.age_sigma = train_ages.std()
            train_ages = self.zscore_normalize_age(train_ages)
            val_ages = self.zscore_normalize_age(val_ages)

        # Zip the conditions into one single list
        train_conditions = [(a, b) for a, b in zip(train_ages, train_sexes)]
        val_conditions = [(a, b) for a, b in zip(val_ages, val_sexes)]

        train_ds = FileListDataset(
            train_images,
            condition_list=train_conditions,
            transform=self.train_transforms,
            data_key=self.data_key,
        )

        val_ds = FileListDataset(
            val_images,
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

    def get_age_balanced_sampler(self, ages):
        # Assuming condition_list holds age information
        age_groups = [age // 10 for age in ages]
        unique_groups, group_counts = np.unique(age_groups, return_counts=True)
        group_weights = {
            group: 1.0 / count for group, count in zip(unique_groups, group_counts)
        }

        # Assign weight to each sample based on its age group
        sample_weights = [group_weights[age_group] for age_group in age_groups]

        # Define a sampler using the sample weights
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # Return a DataLoader with the balanced sampler
        return sampler
