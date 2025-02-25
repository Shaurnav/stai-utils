import os
import numpy as np
import torch
from torchvision.models import resnet50
from tqdm import tqdm

from generative.metrics import FIDMetric
from monai.data import DataLoader, NumpyReader
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


from stai_utils.datasets.dataset_utils import FileListDataset
from stai_utils.evaluations.models.resnet import resnet10
from stai_utils.evaluations.util import create_dataloader_from_dir
from stai_utils.evaluations.metrics.age_regressor import get_ageregressor_model
from stai_utils.evaluations.metrics.sex_classifier import get_sexclassifier_model


class MyReader(NumpyReader):
    def get_data(self, data):
        img_data = data[0]
        img, meta = super().get_data(img_data)
        meta["age"] = data[1]
        meta["sex"] = data[2]
        return img, meta


def _get_medicalnet_model():
    if os.getenv("CLUSTER_NAME") == "haic":
        checkpoint_path = "/hai/scratch/alanqw/models/MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_10.pth"
    elif os.getenv("CLUSTER_NAME") == "sc":
        checkpoint_path = "/simurgh/u/alanqw/models/MedicalNet/MedicalNet_pytorch_files2/pretrain/resnet_10.pth"
    else:
        raise ValueError(
            f"Unknown cluster name: {os.getenv('CLUSTER_NAME')}. Please set the CLUSTER_NAME environment variable correctly."
        )
    res10 = resnet10()
    res10.conv_seg = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten()
    )
    res10 = torch.nn.DataParallel(res10)
    ckpt = torch.load(checkpoint_path)
    res10.load_state_dict(ckpt["state_dict"], strict=False)
    res10.eval()
    return res10


def _get_imagenet_model():
    res50 = resnet50(weights="ResNet50_Weights.DEFAULT")
    res50.eval()
    return res50


def _extract_ageregressor_features_to_dir(
    loader, dest_dir, feature_extractor, device, skip_existing=False
):
    dataset = loader.dataset
    for i in tqdm(range(len(dataset)), desc="Extracting age regressor features"):
        save_path = os.path.join(dest_dir, f"feat_{i}.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping because {save_path} already exists...")
            continue
        data = dataset[i]
        image = torch.tensor(data["vol_data"]).float().to(device)[None]
        age = data["vol_data"].meta["age"]
        sex = data["vol_data"].meta["sex"]
        feat = feature_extractor(image)
        print(feat.shape)

        np.savez(save_path, feat=feat[0].cpu().detach().numpy(), age=age, sex=sex)


def _extract_medicalnet_features_to_dir(
    loader, dest_dir, feature_extractor, device, skip_existing=False
):
    def __itensity_normalize_one_volume__(volume):
        """
        From MedicalNet repository: https://github.com/Tencent/MedicalNet/blob/master/datasets/brains18.py.
        They do data resizing and z-score normalization before feeding the data to the model.

        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = (
            torch.tensor(np.random.normal(0, 1, size=volume.shape))
            .float()
            .to(volume.device)
        )
        out[volume == 0] = out_random[volume == 0]
        return out

    dataset = loader.dataset
    for i in tqdm(range(len(dataset)), desc="Extracting MedicalNet features"):
        save_path = os.path.join(dest_dir, f"feat_{i}.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping because {save_path} already exists...")
            continue
        data = dataset[i]
        image = torch.tensor(data["vol_data"]).float().to(device)[None]
        image = __itensity_normalize_one_volume__(image)
        age = data["vol_data"].meta["age"]
        sex = data["vol_data"].meta["sex"]
        feat = feature_extractor(image)

        np.savez(save_path, feat=feat[0].cpu().detach().numpy(), age=age, sex=sex)


def _extract_imagenet_features_to_dir(
    loader, dest_dir, feature_extractor, device, skip_existing=False
):
    # transforms to convert the input image to the format expected by the model
    def subtract_mean(x: torch.Tensor) -> torch.Tensor:
        """Normalize an input image by subtracting the mean."""
        mean = [0.406, 0.456, 0.485]
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        return x

    def _get_imagenet_features(image, model):
        """Get features from the input image."""
        # If input has just 1 channel, repeat channel to have 3 channels
        if image.shape[1]:
            image = image.repeat(1, 3, 1, 1)

        # Change order from 'RGB' to 'BGR'
        image = image[:, [2, 1, 0], ...]

        # Subtract mean used during training
        image = subtract_mean(image)

        # Get model outputs
        with torch.no_grad():
            feature_image = model(image)

        return feature_image

    dataset = loader.dataset
    for i in tqdm(range(len(dataset)), desc="Extracting ImageNet features"):
        save_path = os.path.join(dest_dir, f"feat_{i}.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping because {save_path} already exists...")
            continue

        data = dataset[i]
        # Convert to 2d
        image = data["vol_data"][None]
        image = image[:, :, image.shape[2] // 2]
        image = torch.tensor(image).float().to(device)
        age = data["vol_data"].meta["age"]
        sex = data["vol_data"].meta["sex"]
        feat = _get_imagenet_features(image, feature_extractor)

        np.savez(save_path, feat=feat[0].cpu().detach().numpy(), age=age, sex=sex)


def evaluate_fid_ageregressor(
    real_img_paths,
    fake_img_paths,
    real_feat_dir,
    fake_feat_dir,
    device,
    skip_existing,
    apply_val_transforms=False,
):
    # Load the medicalnet model
    ageregressor = get_ageregressor_model().encoder.to(device)
    ageregressor = torch.nn.Sequential(
        ageregressor, torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten()
    )

    data_key = "vol_data"
    spacing = (1, 1, 1)
    img_size = (160, 192, 176)
    if apply_val_transforms:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
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
    else:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
            ]
        )

    # Build dataloader from paths
    real_img_loader = DataLoader(
        FileListDataset(
            real_img_paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    fake_img_loader = DataLoader(
        FileListDataset(
            fake_img_paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Extract features from the real and fake samples
    print("Extracting age regressor features...")
    _extract_ageregressor_features_to_dir(
        real_img_loader,
        real_feat_dir,
        ageregressor,
        device,
        skip_existing=skip_existing,
    )
    _extract_ageregressor_features_to_dir(
        fake_img_loader,
        fake_feat_dir,
        ageregressor,
        device,
        skip_existing=skip_existing,
    )

    real_feat_loader = create_dataloader_from_dir(real_feat_dir)
    fake_feat_loader = create_dataloader_from_dir(fake_feat_dir)

    real_feats = []
    fake_feats = []
    for real, fake in zip(real_feat_loader, fake_feat_loader):
        real_feats.append(real["feat"])
        fake_feats.append(fake["feat"])
    return FIDMetric()(torch.vstack(fake_feats), torch.vstack(real_feats)).item()


def evaluate_fid_medicalnet3d(
    real_img_paths,
    fake_img_paths,
    real_feat_dir,
    fake_feat_dir,
    device,
    skip_existing,
    apply_val_transforms=False,
):
    # Load the medicalnet model
    medicalnet = _get_medicalnet_model().to(device)

    data_key = "vol_data"
    spacing = (1, 1, 1)
    img_size = (160, 192, 176)
    if apply_val_transforms:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
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
    else:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
            ]
        )

    # Build dataloader from paths
    real_img_loader = DataLoader(
        FileListDataset(
            real_img_paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    fake_img_loader = DataLoader(
        FileListDataset(
            fake_img_paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Extract features from the real and fake samples
    print("Extracting medicalnet features...")
    _extract_medicalnet_features_to_dir(
        real_img_loader, real_feat_dir, medicalnet, device, skip_existing=skip_existing
    )
    _extract_medicalnet_features_to_dir(
        fake_img_loader, fake_feat_dir, medicalnet, device, skip_existing=skip_existing
    )

    real_feat_loader = create_dataloader_from_dir(real_feat_dir)
    fake_feat_loader = create_dataloader_from_dir(fake_feat_dir)

    real_feats = []
    fake_feats = []
    for real, fake in zip(real_feat_loader, fake_feat_loader):
        real_feats.append(real["feat"])
        fake_feats.append(fake["feat"])
    return FIDMetric()(torch.vstack(fake_feats), torch.vstack(real_feats)).item()


def evaluate_fid_imagenet2d(
    real_img_paths,
    fake_img_paths,
    real_feat_dir,
    fake_feat_dir,
    device,
    skip_existing,
    apply_val_transforms=False,
):
    # Load the imagenet model
    imagenet = _get_imagenet_model().to(device)

    data_key = "vol_data"
    spacing = (1, 1, 1)
    img_size = (160, 192, 176)
    if apply_val_transforms:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
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
    else:
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=[data_key],
                    reader=MyReader(
                        npz_keys=["vol_data", "age", "sex"], channel_dim=None
                    ),
                ),
            ]
        )

    # Build dataloader from paths
    real_img_loader = DataLoader(
        FileListDataset(
            real_img_paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    fake_img_loader = DataLoader(
        FileListDataset(
            fake_img_paths,
            transform=val_transforms,
            data_key="vol_data",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Extract features from the real and fake samples
    print("Extracting imagenet features...")
    _extract_imagenet_features_to_dir(
        real_img_loader, real_feat_dir, imagenet, device, skip_existing=skip_existing
    )
    _extract_imagenet_features_to_dir(
        fake_img_loader, fake_feat_dir, imagenet, device, skip_existing=skip_existing
    )

    real_feat_loader = create_dataloader_from_dir(real_feat_dir)
    fake_feat_loader = create_dataloader_from_dir(fake_feat_dir)

    real_feats = []
    fake_feats = []
    for real, fake in zip(real_feat_loader, fake_feat_loader):
        real_feats.append(real["feat"])
        fake_feats.append(fake["feat"])
    return FIDMetric()(torch.vstack(fake_feats), torch.vstack(real_feats)).item()
