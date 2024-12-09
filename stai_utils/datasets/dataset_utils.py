import pickle
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

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
)


def get_t1_all_file_list(zscore=False):
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
    val_images = []
    val_ages = []

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

            # Append new data to existing lists
            if not train_images:  # More Pythonic way to check if the list is empty
                # Direct assignment for the first file
                train_images = train_new_images
                train_ages = train_new_ages
                train_sex = train_new_sex

                val_images = val_new_images
                val_ages = val_new_ages
                val_sex = val_new_sex
            else:
                # Concatenation for subsequent files
                train_images += train_new_images
                train_ages += train_new_ages
                train_sex += train_new_sex

                val_images += val_new_images
                val_ages += val_new_ages
                val_sex += val_new_sex

            # Debug output to check the results
            print(train_images[-1])  # Print the last path

    train_ages = np.array(train_ages)
    val_ages = np.array(val_ages)
    if zscore:
        # Z-score normalization for age
        mu = train_ages.mean()
        sigma = train_ages.std()
        train_ages = (train_ages - mu) / (sigma + 1e-8)
        val_ages = (val_ages - mu) / (sigma + 1e-8)

    train_images = [file_dir_prefix + train_image for train_image in train_images]
    val_images = [file_dir_prefix + val_image for val_image in val_images]

    print(len(train_images))
    print(len(val_images))

    # Zip the conditions into one single list
    train_conditions = [(a, b) for a, b in zip(train_ages, train_sex)]
    val_conditions = [(a, b) for a, b in zip(val_ages, val_sex)]

    return train_images, train_conditions, val_images, val_conditions


def prepare_dataloader_from_list(
    batch_size,
    patch_size,
    randcrop=True,
    rank=0,
    world_size=1,
    size_divisible=16,
    channel=0,
    spacing=(1.0, 1.0, 1.0),
):
    ddp_bool = world_size > 1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    if randcrop:
        train_crop_transform = RandSpatialCropd(
            keys=["image"], roi_size=patch_size, random_size=False
        )
        val_patch_size = [
            int(np.ceil(1.5 * p / size_divisible) * size_divisible) for p in patch_size
        ]
    else:
        train_crop_transform = CenterSpatialCropd(keys=["image"], roi_size=patch_size)
        val_patch_size = patch_size

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
            train_crop_transform,
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )

    train_images, train_conditions, val_images, val_conditions = get_t1_all_file_list()

    train_ds = FileListDataset(
        train_images,
        condition_list=train_conditions,
        transform=train_transforms,
    )

    val_ds = FileListDataset(
        val_images,
        condition_list=val_conditions,
        transform=val_transforms,
    )

    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank
        )
    else:
        train_sampler = None
        val_sampler = None

    print("shuffle for train: ", (not ddp_bool))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(not ddp_bool),
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler,
    )
    if rank == 0:
        # print(f'Image shape {train_ds[0]["image"]}')
        print(f'Train Image shape {train_ds[0]["image"].shape}')
        print(f'Val Image shape {val_ds[0]["image"].shape}')
    return train_loader, val_loader


class FileListDataset(Dataset):
    def __init__(
        self,
        file_list,
        condition_list=None,
        transform=None,
    ):
        self.file_list = file_list
        self.transform = transform
        self.condition_list = condition_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = {"image": img_path}

        if self.transform:
            data = self.transform(data)

        condition_tensor = self.condition_list[idx]
        data["age"] = condition_tensor[0]
        data["sex"] = condition_tensor[1]
        return data
