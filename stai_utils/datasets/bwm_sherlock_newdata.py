import pickle
import numpy as np
import os
import torch
import math
import re
import pandas as pd
import random
from pathlib import Path
import nibabel as nib  # For loading .nii.gz files
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

import matplotlib.pyplot as plt
from tqdm import tqdm

MODALITY_MAP = {"t1": 0.0, "t2": 1.0}

def generate_paths_and_info_pkl_from_csv(
    csv_file_path, 
    output_pkl_path, 
    image_column="MNI_Z_Cropped", 
    age_column="age", 
    sex_column="sex",
    train_split=0.99,
    random_seed=42,
    verbose=True
):
    """
    Generate paths_and_info_relpath.pkl file from CSV data.
    
    Args:
        csv_file_path (str): Path to the CSV file containing metadata
        output_pkl_path (str): Path where to save the pickle file
        image_column (str): Column name containing image file paths
        age_column (str): Column name containing age values
        sex_column (str): Column name containing sex values (should be 1/2 for male/female)
        train_split (float): Fraction of data to use for training (0.8 = 80% train, 20% val)
        random_seed (int): Random seed for reproducible splits
        verbose (bool): Whether to print progress information
    
    Returns:
        dict: The generated data dictionary with train/val splits
    """
    if verbose:
        print(f"Processing CSV file: {csv_file_path}")
    
    # Read CSV file
    print("csv_file_path", csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(df.columns)
    if 'research_group' in df.columns:
        print(df['research_group'].value_counts())
    
    if verbose:
        print(f"Loaded {len(df)} rows from CSV")
        print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_columns = [image_column, age_column, sex_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle research_group column - create with value 0 if not present
    if 'research_group' not in df.columns:
        df['research_group'] = 0
        if verbose:
            print("Created 'research_group' column with value 0")
    
    # Clean and filter data
    # Remove rows with missing values
    df_clean = df.dropna(subset=[image_column, age_column, sex_column])
    
    # Convert sex to 0/1 (assuming 1=male, 2=female in original data)
    df_clean['sex_binary'] = df_clean[sex_column].map({1: 0, 2: 1})  # 1->0 (male), 2->1 (female)
    # print(len(df_clean), -1)

    # Filter out invalid sex values
    df_clean = df_clean[df_clean['sex_binary'].isin([0, 1])]
    # print(len(df_clean), 0)
    
    # Convert age to float
    df_clean['age_float'] = pd.to_numeric(df_clean[age_column], errors='coerce')
    df_clean = df_clean.dropna(subset=['age_float'])
    # print(len(df_clean), 1)
    
    # Filter out invalid ages (e.g., negative or very high ages)
    df_clean = df_clean[(df_clean['age_float'] > 0) & (df_clean['age_float'] < 120)]
    # print(len(df_clean), 2)

    if verbose:
        print(f"After cleaning: {len(df_clean)} valid rows")
        print(f"Age range: {df_clean['age_float'].min():.1f} - {df_clean['age_float'].max():.1f}")
        print(f"Sex distribution: {df_clean['sex_binary'].value_counts().to_dict()}")
        print(f"Research group distribution: {df_clean['research_group'].value_counts().to_dict()}")
    
    # Convert absolute paths to relative paths
    # Use the mabbasi directory as the base directory for relative paths
    base_dir = "/simurgh/group/BWM/DataSets"
    
    if len(df_clean) > 0:
        df_clean['relative_path'] = df_clean[image_column].apply(
            lambda x: str(Path(x.replace('/simurgh/group/BWM/DataSets', base_dir)).relative_to(base_dir)) if x.startswith('/') else x
        )
    else:
        df_clean['relative_path'] = df_clean[image_column]
    

    
    # Split into train/val
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the data
    df_shuffled = df_clean.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    train_size = int(len(df_shuffled) * train_split)
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:]
    
    if verbose:
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create the data structure expected by BWMSherlock
    data = {
        'train': {
            'paths': train_df['relative_path'].tolist(),
            'ages': train_df['age_float'].tolist(),
            'sexes': train_df['sex_binary'].tolist(),
            'research_group': train_df['research_group'].tolist(),
        },
        'val': {
            'paths': val_df['relative_path'].tolist(),
            'ages': val_df['age_float'].tolist(),
            'sexes': val_df['sex_binary'].tolist(),
            'research_group': val_df['research_group'].tolist(),
        }
    }
    
    # Save to pickle file
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        print(f"Saved pickle file to: {output_pkl_path}")
        print(f"Train paths: {len(data['train']['paths'])}")
        print(f"Val paths: {len(data['val']['paths'])}")
    
    return data


def generate_all_dataset_pickles(
    base_dir="/hai/scratch/mabbasi/BWM/DataSets",
    output_dir="/hai/scratch/trangn/BWM/DataSets",
    train_split=0.95,
    random_seed=42,
    verbose=True
):
    """
    Generate pickle files for all datasets in the BWM DataSets directory.
    
    Args:
        base_dir (str): Base directory containing dataset folders
        output_dir (str): Output directory for pickle files
        train_split (float): Fraction of data to use for training
        random_seed (int): Random seed for reproducible splits
        verbose (bool): Whether to print progress information
    """
    datasets = ['ABCD', 'ADNI', 'AIBL', 'HCP-Aging', 'HCP-Development', 'HCP-YA', 'OpenNeuro', 'PPMI']
    
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
        if not os.path.exists(dataset_dir):
            if verbose:
                print(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Use the first CSV file found
        csv_file = os.path.join(dataset_dir, f'processed/Structural/metadata/{dataset}_T1_metadata.csv')
        if dataset == 'HCP-Development':
            csv_file = os.path.join(dataset_dir, f'processed/Structural/metadata/HCP-Dev_T1_metadata.csv')
        if dataset == 'HCP-Aging':
            csv_file = os.path.join(dataset_dir, f'processed/Structural/metadata/HCP-Aging_T1_metadata.csv')
        if dataset == 'OpenNeuro':
            csv_file = os.path.join(dataset_dir, f'processed/Structural/metadata/T1_metadata.csv')
        if dataset == 'HCP-YA':
            csv_file = os.path.join(dataset_dir, f'processed/Structural/metadata/T1w_MPR1_metadata.csv')

        if not os.path.exists(csv_file):
            print(f"No metadata CSV files found in {dataset_dir}, {csv_file}")
        output_pkl = os.path.join(output_dir, dataset, "paths_and_info_relpath.pkl")
        
        if verbose:
            print(f"\nProcessing dataset: {dataset}")
            print(f"CSV file: {csv_file}")
            print(f"Output pickle: {output_pkl}")
        
        try:
            generate_paths_and_info_pkl_from_csv(
                csv_file_path=csv_file,
                output_pkl_path=output_pkl,
                train_split=train_split,
                random_seed=random_seed,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"Error processing {dataset}: {e}")


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
        root_dir = "/simurgh/group/BWM/DataSets/"

    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        # Joy
        t1_dataset_names = [
            "/simurgh/u/joyghosh/BWM/DataSets/ABCD/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/ADNI/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/AIBL/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/HCP-Aging/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/HCP-Development/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/HCP-YA/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/OpenNeuro/paths_and_info_relpath.pkl",
            "/simurgh/u/joyghosh/BWM/DataSets/PPMI/paths_and_info_relpath.pkl",
        ]
        t2_dataset_names = [
        ]

    elif cluster_name == "haic":
        t1_dataset_names = [
            "/hai/scratch/trangn/BWM/DataSets/ABCD/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/ADNI/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/AIBL/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/HCP-Aging/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/HCP-Development/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/HCP-YA/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/OpenNeuro/paths_and_info_relpath.pkl",
            "/hai/scratch/trangn/BWM/DataSets/PPMI/paths_and_info_relpath.pkl",
        ]
        t2_dataset_names = [
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


                # HEALTHY control filter
                if "ADNI" in dataset_name:
                    data["train"] = filter_dict_by_key(data["train"], "research_group", "CN")
                    data["val"]   = filter_dict_by_key(data["val"], "research_group", "CN")

                if "PPMI" in dataset_name: 
                    data["train"] = filter_dict_by_key(data["train"], "research_group", "Control")
                    data["val"]   = filter_dict_by_key(data["val"], "research_group", "Control")



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

    # dataset_name_ = dataset_name.split('/DataSets/')[1].split('/')[0] 
    # print(dataset_name_)
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

        # Convert file_list to a list if it contains tuples, and extract file paths
        processed_file_list = []
        for i, item in enumerate(self.file_list):
            if isinstance(item, tuple):
                # If it's a tuple, extract the file path (first element)
                file_path = item[0]
                print(file_path)
                # Modify the file path
                modified_path = os.path.join(os.path.dirname(file_path), "T1w_mni_zscore_fixed_cropped.nii.gz")
            else:
                # If it's already a string, process it directly
                modified_path = os.path.join(os.path.dirname(item), "T1w_mni_zscore_fixed_cropped.nii.gz")
            
            # TODO this line takes a very long time? I wonder why
            if os.path.exists(modified_path):
                processed_file_list.append(modified_path)
        
        self.file_list = processed_file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        # print("img_path", img_path)
        data = {self.data_key: img_path}
        
        condition_tensor = self.condition_list[idx]
        data["age"] = condition_tensor[0]
        data["sex"] = condition_tensor[1]
        data["modality"] = condition_tensor[2]
        data["path"] = img_path
        # print(data['age'], data['sex'], img_path)
        if self.transform:
            data = self.transform(data)

        return data


class BWMSherlock:
    def __init__(
        self,
        img_size,
        num_workers,
        read_from_scr=False,
        modality=("t1"),
        age_normalization=None,
        rank=0,
        world_size=1,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=False,
        max_sample = None,
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

        # print("img_size", img_size)

        self.train_transforms = Compose(
            [
                LoadImaged(keys=[data_key]),
                EnsureChannelFirstd(keys=[data_key]),
                # Lambdad(keys=data_key, func=lambda x: x[0, :, :, :]),
                # EnsureChannelFirstd(keys=[data_key], channel_dim=0),
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
                # Lambdad(keys=data_key, func=lambda x: x[0, :, :, :]),
                # EnsureChannelFirstd(keys=[data_key], channel_dim=0),
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

    def get_dataloaders(self, batch_size, drop_last=False, get_val=False):
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

        if get_val:
            return val_paths, val_conditions

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

        print(f"this is the length of VAL DS: {val_ds}")
        
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

    def __len__(self):
        # Return the number of training samples
        train_data, _ = get_all_file_list_bwm_sherlock(
            read_from_scr=self.read_from_scr, modality=self.modality
        )
        return len(train_data)


if __name__ == "__main__":
    print("="*60)
    print("BWM SHERLOCK DATASET TESTING")
    print("="*60)
    
    # Check if pickle files already exist
    output_dir = "/simurgh/u/joyghosh/BWM/DataSets"
    datasets = ['ABCD', 'ADNI', 'AIBL', 'HCP-Aging', 'HCP-Development', 'HCP-YA', 'OpenNeuro', 'PPMI']
    
    pickle_files_exist = True
    missing_datasets = []
    
    print("Checking for existing pickle files...")
    for dataset in datasets:
        pickle_file = os.path.join(output_dir, dataset, "paths_and_info_relpath.pkl")
        if os.path.exists(pickle_file):
            print(f"✓ {dataset} pickle file exists")
        else:
            print(f"✗ {dataset} pickle file missing")
            pickle_files_exist = False
            missing_datasets.append(dataset)
    
    # Generate pickle files if they don't exist
    if not pickle_files_exist:
        print(f"\nGenerating missing pickle files for: {missing_datasets}")
        try:
            generate_all_dataset_pickles(
                base_dir="/simurgh/group/BWM/DataSets",
                output_dir=output_dir,
                verbose=True
            )
            print("✓ Successfully generated all dataset pickle files")
        except Exception as e:
            print(f"✗ Error generating pickle files: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✓ All pickle files exist, proceeding to load dataset")
    
    # Initialize the BWMSherlock class
    print("\n" + "="*50)
    print("LOADING BWMSHERLOCK DATASET")
    print("="*50)
    
    dataset = BWMSherlock(
        img_size=(160, 192, 176),
        num_workers=1,
        read_from_scr=False,
        modality=("t1",),
        age_normalization=None,
        rank=0,
        world_size=1,
        spacing=(1.0, 1.0, 1.0),
        data_key="image",
        sample_balanced_age_for_training=True, 
    )

    print("Dataset initialized successfully")
    
    # Get dataloaders
    print("\n" + "="*50)
    print("GETTING DATALOADERS")
    print("="*50)
    
    try:
        train_loader, val_loader = dataset.get_dataloaders(batch_size=32, drop_last=False)
        
        print(f"\nTrain loader length: {len(train_loader)}")
        print(f"Validation loader length: {len(val_loader)}")
        
        # Print sample data from training set
        print("\n" + "="*50)
        print("SAMPLE TRAINING DATA")
        print("="*50)
        
        for i, batch in enumerate(train_loader):
            print(f"\nTraining Batch {i+1}:")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Image dtype: {batch['image'].dtype}")
            print(f"  Image min/max: {batch['image'].min():.4f}/{batch['image'].max():.4f}")
            print(f"  Image path: {batch['path']}")
            print(f"  Ages: {batch['age']}")
            print(f"  Sexes: {batch['sex']}")
            print(f"  Modalities: {batch['modality']}")
            
            # Print first few file paths if available
            if hasattr(train_loader.dataset, 'file_list'):
                print(f"  Sample file paths:")
                for j in range(min(2, len(train_loader.dataset.file_list))):
                    print(f"    {j+1}: {train_loader.dataset.file_list[j]}")
            
            # Get the first image, age, and sex in the batch.
            img = batch['image'][0].numpy()  # shape: (C, H, W, D) or (1, H, W, D)
            if img.ndim == 4:
                img = img[0]  # remove channel dimension if present

            age = batch['age'][0].item() if hasattr(batch['age'][0], 'item') else batch['age'][0]
            sex = batch['sex'][0].item() if hasattr(batch['sex'][0], 'item') else batch['sex'][0]

            # Get the center slices for each view
            axial_slice = img[:, :, img.shape[2] // 2]
            sagittal_slice = img[img.shape[0] // 2, :, :]
            coronal_slice = img[:, img.shape[1] // 2, :]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')
            axes[0].set_title('Axial')
            axes[1].imshow(sagittal_slice.T, cmap='gray', origin='lower')
            axes[1].set_title('Sagittal')
            axes[2].imshow(coronal_slice.T, cmap='gray', origin='lower')
            axes[2].set_title('Coronal')

            # Add age and sex as a super title
            sex_str = "Male" if sex == 0 else "Female"
            fig.suptitle(f"Age: {age:.1f}, Sex: {sex_str}", fontsize=16)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = f"sample_batch_{i+1}.png"
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved sample visualization to: {out_path}")
            
            if i >= 2:  # Only print first 3 batches
                break
        
        # Print sample data from validation set
        print("\n" + "="*50)
        print("SAMPLE VALIDATION DATA")
        print("="*50)
        
        for i, batch in enumerate(val_loader):
            print(f"\nValidation Batch {i+1}:")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Image dtype: {batch['image'].dtype}")
            print(f"  Image min/max: {batch['image'].min():.4f}/{batch['image'].max():.4f}")
            print(f"  Ages: {batch['age']}")
            print(f"  Sexes: {batch['sex']}")
            print(f"  Modalities: {batch['modality']}")
            
            # Print first few file paths if available
            if hasattr(val_loader.dataset, 'file_list'):
                print(f"  Sample file paths:")
                for j in range(min(2, len(val_loader.dataset.file_list))):
                    print(f"    {j+1}: {val_loader.dataset.file_list[j]}")
            
            if i >= 2:  # Only print first 3 batches
                break
                
        print("\n" + "="*50)
        print("SAMPLE DATA PRINTING COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"Error getting dataloaders: {e}")
        import traceback
        traceback.print_exc()

