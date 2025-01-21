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
    for p, a, s, m in l:
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
        qc.append((p, a_val, s_val, m_val))

    return qc


def get_all_file_list_bwm_sherlock(modality=("t1", "t2"), verbose=True):
    """Returns file list for data in BWM Sherlock directory."""
    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        PREFIX_MAP = {
            "/scratch/groups/eadeli/data/stru_new/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
            "proc/t1/hcp_dev_t1/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/hcp_dev/",
            "/simurgh/group/BWM/Sherlock/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
            #####
            "/scratch/groups/eadeli/data/stru_new/t2/fully_proc/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/",
            "proc/t2/abcd_t2/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/abcd/",
            "proc/t2/hcp_ag_t2/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/hcp_ag/",
            "/simurgh/group/BWM/Sherlock/t2/fully_proc/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/",
        }
        t1_dataset_names = [
            "/simurgh/group/BWM/Sherlock/t1/metadata/abcd/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/adni/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_ag/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_dev/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_ya_hcp1200/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/ppmi/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t1/metadata/opne_ds004215/paths_and_info.pkl",
        ]
        t2_dataset_names = [
            "/simurgh/group/BWM/Sherlock/t2/metadata/ppmi/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t2/metadata/opne_ds004215/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_ya_hcp1200/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info.pkl",
            "/simurgh/group/BWM/Sherlock/t2/metadata/abcd/paths_and_info.pkl",
            # "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_dev/paths_and_info.pkl",
        ]
    elif cluster_name == "haic":
        PREFIX_MAP = {
            # "/scratch/groups/eadeli/data/stru_new/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
            # "proc/t1/hcp_dev_t1/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/hcp_dev/",
            # "/simurgh/group/BWM/Sherlock/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
            #####
            "/scratch/groups/eadeli/data/stru_new/t2/fully_proc/": "/hai/scratch/alanqw/BWM/Sherlock/t2/fully_proc/",
            "proc/t2/abcd_t2/": "/hai/scratch/alanqw/BWM/Sherlock/t2/fully_proc/abcd/",
            "proc/t2/hcp_ag_t2/": "/hai/scratch/alanqw/BWM/Sherlock/t2/fully_proc/hcp_ag/",
            "/simurgh/group/BWM/Sherlock/t2/fully_proc/": "/hai/scratch/alanqw/BWM/Sherlock/t2/fully_proc/",
        }
        t1_dataset_names = [
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/abcd/paths_and_info_flexpath.pkl",
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/adni_t1/paths_and_info_flexpath.pkl",
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_ag_t1/paths_and_info_flexpath.pkl",
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_dev_t1/paths_and_info_flexpath.pkl",
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_ya_mpr1/paths_and_info_flexpath.pkl",
            "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/metadata/ppmi_t1/paths_and_info_flexpath.pkl",
        ]
        t2_dataset_names = [
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/ppmi/paths_and_info.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/opne_ds004215/paths_and_info.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ya_hcp1200/paths_and_info.pkl",
            # "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info_w_mninonlinear.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/abcd/paths_and_info.pkl",
            # "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_dev/paths_and_info.pkl",
        ]
    # elif cluster_name == "sherlock":
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )

    def fix_paths(old_paths):
        new_paths = []
        for path in old_paths:
            # Find the prefix that matches the path
            old_prefix = [p for p in tuple(PREFIX_MAP.keys()) if path.startswith(p)]
            if len(old_prefix) == 0:
                new_paths.append(path)  # No match found, don't change the path
            elif len(old_prefix) == 1:
                old_prefix = old_prefix[0]
                new_prefix = PREFIX_MAP[old_prefix]
                updated_path = path.replace(old_prefix, new_prefix, 1)
                new_paths.append(updated_path)
            else:
                raise ValueError("Multiple matches found for path:", path)

        return new_paths

    if isinstance(modality, str):
        modality = [modality]

    train_paths = []
    train_ages = []
    train_sexes = []
    train_modalities = []
    val_paths = []
    val_ages = []
    val_sexes = []
    val_modalities = []

    for m in modality:
        if m == "t1":
            dataset_names = t1_dataset_names
        elif m == "t2":
            dataset_names = t2_dataset_names
        else:
            raise ValueError(f"Unknown modality: {m}")
        for dataset_name in dataset_names:
            with open(dataset_name, "rb") as file:
                data = pickle.load(file)

                age_key = "ages" if "ages" in data["train"].keys() else "age"
                sex_key = "sexes" if "sexes" in data["train"].keys() else "sex"

                # Convert paths and ages to lists if they are NumPy arrays
                dataset_train_paths = data["train"]["paths"].tolist()
                dataset_train_ages = data["train"][age_key].tolist()
                dataset_train_sexes = data["train"][sex_key].tolist()

                dataset_val_paths = data["val"]["paths"].tolist()
                dataset_val_ages = data["val"][age_key].tolist()
                dataset_val_sexes = data["val"][sex_key].tolist()

                assert (
                    len(dataset_train_paths)
                    == len(dataset_train_ages)
                    == len(dataset_train_sexes)
                )
                assert (
                    len(dataset_val_paths)
                    == len(dataset_val_ages)
                    == len(dataset_val_sexes)
                )
                if verbose:
                    print("Dataset:", dataset_name)
                    print("Keys:", data["train"].keys())
                    # for key in data['train'].keys():
                    #     print(f'\n{key}')
                    #     print(data['train'][key])
                    print(f"-> Num train: {len(dataset_train_paths)}")
                    print(f"-> Num val: {len(dataset_val_paths)}")

                train_paths += dataset_train_paths
                train_ages += dataset_train_ages
                train_sexes += dataset_train_sexes
                train_modalities += [MODALITY_MAP[m]] * len(dataset_train_paths)

                val_paths += dataset_val_paths
                val_ages += dataset_val_ages
                val_sexes += dataset_val_sexes
                val_modalities += [MODALITY_MAP[m]] * len(dataset_val_paths)

        if m == "t1":  # TODO: remove this
            train_paths = [
                "/hai/scratch/fangruih/data/" + train_path for train_path in train_paths
            ]
            val_paths = [
                "/hai/scratch/fangruih/data/" + val_path for val_path in val_paths
            ]

        train_paths = fix_paths(train_paths)
        val_paths = fix_paths(val_paths)

    assert (
        len(train_paths) == len(train_ages) == len(train_sexes) == len(train_modalities)
    )
    assert len(val_paths) == len(val_ages) == len(val_sexes) == len(val_modalities)

    train_data = list(zip(train_paths, train_ages, train_sexes, train_modalities))
    val_data = list(zip(val_paths, val_ages, val_sexes, val_modalities))

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
        data["age"] = condition_tensor[0]
        data["sex"] = condition_tensor[1]
        data["modality"] = condition_tensor[2]
        return data


class BWMSherlock:
    def __init__(
        self,
        img_size,
        num_workers,
        modality=("t1", "t2"),
        zscore_age=False,
        rank=0,
        world_size=1,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=False,
    ):
        self.num_workers = num_workers
        self.modality = modality
        self.zscore_age = zscore_age
        self.rank = rank
        self.world_size = world_size
        self.ddp_bool = world_size > 1
        self.data_key = data_key
        self.sample_balanced_age_for_training = sample_balanced_age_for_training

        self.age_mu = 0
        self.age_sigma = 1

        self.train_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key]),
                Lambdad(keys=data_key, func=lambda x: x[0, :, :, :]),
                EnsureChannelFirstd(keys=[data_key], channel_dim=0),
                EnsureTyped(keys=[data_key]),
                Orientationd(keys=[data_key], axcodes="RAS"),
                Spacingd(keys=[data_key], pixdim=spacing, mode=("bilinear")),
                ResizeWithPadOrCropd(keys=[data_key], spatial_size=img_size),
                # train_crop_transform,
                ScaleIntensityRangePercentilesd(
                    keys=data_key, lower=0, upper=100, b_min=0, b_max=1
                ),
                EnsureTyped(keys=data_key, dtype=torch.float32),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key]),
                Lambdad(keys=data_key, func=lambda x: x[0, :, :, :]),
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
    ):
        train_data, val_data = get_all_file_list_bwm_sherlock(self.modality)

        train_paths, train_ages, train_sexes, train_modalities = zip(*train_data)
        val_paths, val_ages, val_sexes, val_modalities = zip(*val_data)

        train_ages = np.array(train_ages)
        val_ages = np.array(val_ages)
        if self.zscore_age:
            # Z-score normalization for age
            self.age_mu = np.mean(train_ages)
            self.age_sigma = np.std(train_ages)
            train_ages = self.zscore_normalize_age(train_ages)
            val_ages = self.zscore_normalize_age(val_ages)

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
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=val_sampler,
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


if __name__ == "__main__":
    dataset = BWMSherlock(
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
