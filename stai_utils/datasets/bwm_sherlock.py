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


def get_all_file_list_bwm_sherlock(
    read_from_scr=False, modality=("t1", "t2"), verbose=True
):
    """Returns file list for data in BWM Sherlock directory."""
    if read_from_scr:
        root_dir = "/scr/alanqw/BWM/Sherlock/"
    else:
        root_dir = "/hai/scratch/alanqw/BWM/Sherlock/"

    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        raise NotImplementedError
        # PREFIX_MAP = {
        #     "/scratch/groups/eadeli/data/stru_new/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
        #     "proc/t1/hcp_dev_t1/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/hcp_dev/",
        #     "/simurgh/group/BWM/Sherlock/t1/fully_proc/": "/simurgh/group/BWM/Sherlock/t1/fully_proc/",
        #     #####
        #     "/scratch/groups/eadeli/data/stru_new/t2/fully_proc/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/",
        #     "proc/t2/abcd_t2/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/abcd/",
        #     "proc/t2/hcp_ag_t2/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/hcp_ag/",
        #     "/simurgh/group/BWM/Sherlock/t2/fully_proc/": "/simurgh/group/BWM/Sherlock/t2/fully_proc/",
        # }
        # t1_dataset_names = [
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/abcd/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/adni/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_ag/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_dev/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/hcp_ya_hcp1200/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/ppmi/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t1/metadata/opne_ds004215/paths_and_info.pkl",
        # ]
        # t2_dataset_names = [
        #     "/simurgh/group/BWM/Sherlock/t2/metadata/ppmi/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t2/metadata/opne_ds004215/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_ya_hcp1200/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info.pkl",
        #     "/simurgh/group/BWM/Sherlock/t2/metadata/abcd/paths_and_info.pkl",
        #     # "/simurgh/group/BWM/Sherlock/t2/metadata/hcp_dev/paths_and_info.pkl",
        # ]
    elif cluster_name == "haic":
        t1_dataset_names = [
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/abcd/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/adni/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/hcp_ag/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/hcp_dev/paths_and_info_relpath_w_mninonlinear.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/hcp_ya_hcp1200/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/opne_ds004215/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t1/metadata/ppmi/paths_and_info_relpath.pkl",
        ]
        t2_dataset_names = [
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/ppmi/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/opne_ds004215/paths_and_info_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ya_hcp1200/paths_and_info_relpath.pkl",
            # "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_ag/paths_and_info_w_mninonlinear_relpath.pkl",
            "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/abcd/paths_and_info_relpath.pkl",
            # "/hai/scratch/alanqw/BWM/Sherlock/t2/metadata/hcp_dev/paths_and_info.pkl",
        ]
    elif cluster_name == "sherlock":
        raise NotImplementedError
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )

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

                def filter_dict_by_key(data, filter_key, filter_value):
                    """
                    Filters all lists in a dictionary based on whether the entries in a specific key match a given value.

                    Parameters:
                    - data (dict): A dictionary where all values are lists of the same length.
                    - filter_key (str): The key whose values will be used for filtering.
                    - filter_value (any): The value to filter by.

                    Returns:
                    - dict: A new dictionary with only the filtered entries.
                    """
                    if filter_key not in data:
                        raise KeyError(
                            f"Key '{filter_key}' not found in the dictionary."
                        )

                    # Boolean mask (True where the value matches filter_value)
                    mask = [val == filter_value for val in data[filter_key]]

                    # Filter each list using list comprehension to handle varying data types
                    return {
                        k: [v[i] for i in range(len(v)) if mask[i]]
                        for k, v in data.items()
                    }

                if "adni" in dataset_name:
                    data["train"] = filter_dict_by_key(data["train"], "groups", "CN")
                    data["val"] = filter_dict_by_key(data["val"], "groups", "CN")

                if "ppmi" in dataset_name:
                    data["train"] = filter_dict_by_key(
                        data["train"], "groups", "Control"
                    )
                    data["val"] = filter_dict_by_key(data["val"], "groups", "Control")

                age_key = "ages" if "ages" in data["train"].keys() else "age"
                sex_key = "sexes" if "sexes" in data["train"].keys() else "sex"

                # Convert paths and ages to lists if they are NumPy arrays
                dataset_train_paths = data["train"]["paths"]
                if isinstance(dataset_train_paths, np.ndarray):
                    dataset_train_paths = dataset_train_paths.tolist()
                dataset_train_ages = data["train"][age_key]
                if isinstance(dataset_train_ages, np.ndarray):
                    dataset_train_ages = dataset_train_ages.tolist()
                dataset_train_sexes = data["train"][sex_key]
                if isinstance(dataset_train_sexes, np.ndarray):
                    dataset_train_sexes = dataset_train_sexes.tolist()

                dataset_val_paths = data["val"]["paths"]
                if isinstance(dataset_val_paths, np.ndarray):
                    dataset_val_paths = dataset_val_paths.tolist()
                dataset_val_ages = data["val"][age_key]
                if isinstance(dataset_val_ages, np.ndarray):
                    dataset_val_ages = dataset_val_ages.tolist()
                dataset_val_sexes = data["val"][sex_key]
                if isinstance(dataset_val_sexes, np.ndarray):
                    dataset_val_sexes = dataset_val_sexes.tolist()

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

    assert (
        len(train_paths) == len(train_ages) == len(train_sexes) == len(train_modalities)
    )
    assert len(val_paths) == len(val_ages) == len(val_sexes) == len(val_modalities)

    # Prepend root file path
    train_paths = [os.path.join(root_dir, p) for p in train_paths]
    val_paths = [os.path.join(root_dir, p) for p in val_paths]

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
        try:
            data = {self.data_key: img_path}

            if self.transform:
                data = self.transform(data)

            condition_tensor = self.condition_list[idx]
            data["img_path"] = img_path
            data["age"] = condition_tensor[0]
            data["sex"] = condition_tensor[1]
            data["modality"] = condition_tensor[2]
            return data
        except Exception as e:
            print(f"Error loading file {img_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            # Try to get file size and shape info
            try:
                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path)
                    print(f"File size: {file_size} bytes")
                    # Try to load the file directly to check its shape
                    try:
                        arr = np.load(img_path, allow_pickle=True)
                        print(f"Array shape: {arr.shape}")
                        print(f"Array size: {arr.size}")
                    except Exception as load_error:
                        print(f"Could not load file directly: {str(load_error)}")
                else:
                    print("File does not exist")
            except Exception as info_error:
                print(f"Could not get file info: {str(info_error)}")
            raise  # Re-raise the original exception


class BWMSherlock:
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
                    keys=data_key, lower=0, upper=99.5, b_min=0, b_max=1
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

    @staticmethod
    def _zscore_normalize(x, mu, sigma):
        return (x - mu) / (sigma + 1e-8)

    @staticmethod
    def _inverse_zscore_normalize(x, mu, sigma):
        return x * sigma + mu

    @staticmethod
    def _min_max_scale(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val)

    @staticmethod
    def _inverse_min_max_scale(x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def normalize_age(self, ages):
        if self.age_normalization == "zscore":
            ages = self._zscore_normalize(ages, self.age_mu, self.age_sigma)
        elif self.age_normalization == "min-max":
            ages = self._min_max_scale(ages, self.age_min, self.age_max)
        return ages

    def unnormalize_age(self, ages):
        if self.age_normalization == "zscore":
            ages = self._inverse_zscore_normalize(ages, self.age_mu, self.age_sigma)
        elif self.age_normalization == "min-max":
            ages = self._inverse_min_max_scale(ages, self.age_min, self.age_max)
        return ages

    def get_dataloaders(self, batch_size, drop_last=False):
        train_data, val_data = get_all_file_list_bwm_sherlock(
            read_from_scr=self.read_from_scr, modality=self.modality
        )

        train_paths, train_ages, train_sexes, train_modalities = zip(*train_data)
        val_paths, val_ages, val_sexes, val_modalities = zip(*val_data)

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
