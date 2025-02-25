import torch
from generative.metrics import MultiScaleSSIMMetric
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


class MyReader(NumpyReader):
    def get_data(self, data):
        img_data = data[0]
        img, meta = super().get_data(img_data)
        meta["age"] = data[1]
        meta["sex"] = data[2]
        return img, meta


def compute_pairwise_msssim(paths, N=1000, apply_val_transforms=False):
    # Build dataloader from paths
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

    loader = DataLoader(
        FileListDataset(
            paths,
            transform=val_transforms,
            data_key=data_key,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    tot_metric = 0
    count = 0
    msssim = MultiScaleSSIMMetric(spatial_dims=3, kernel_size=9)
    for i, data1 in enumerate(loader):
        for j, data2 in enumerate(loader):
            if count >= N:
                break
            if i != j:
                img1 = data1["vol_data"].float()
                img2 = data2["vol_data"].float()
                tot_metric += msssim(img1, img2).item()
                count += 1

    return tot_metric / count
