import os
import torch
from tqdm import tqdm

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
from stai_utils.evaluations.models.unet3d.model import UNet3D
from stai_utils.evaluations.models.finetune_model import FinetuneModel


def get_model(checkpoint_path):
    encoder = UNet3D(
        1,
        1,
        final_sigmoid=False,
        f_maps=32,  # Used by nnUNet
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=False,
        conv_padding=1,
        use_checkpoint=False,
    )
    encoder.decoders = torch.nn.ModuleList([])
    encoder.final_conv = torch.nn.Identity()
    model = FinetuneModel(
        encoder,
        output_dim=1,
        dim=3,
        use_checkpoint=False,
    )
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model


class MyReader(NumpyReader):
    def get_data(self, data):
        img_data = data[0]
        img, meta = super().get_data(img_data)
        meta["age"] = data[1]
        meta["sex"] = data[2]
        return img, meta


def evaluate_age_regression(paths, args):
    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "haic":
        checkpoint_path = (
            "/hai/scratch/alanqw/models/age_regressor/epoch148_trained_model.pth.tar"
        )
    elif cluster_name == "sc":
        checkpoint_path = (
            "/simurgh/u/alanqw/models/age_regressor/epoch148_trained_model.pth.tar"
        )
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )
    model = get_model(checkpoint_path)
    model.to(args.device)
    model.eval()

    data_key = "vol_data"
    spacing = (1, 1, 1)
    img_size = (160, 192, 176)
    val_transforms = Compose(
        [
            LoadImaged(
                keys=[data_key],
                reader=MyReader(npz_keys=["vol_data", "age", "sex"], channel_dim=None),
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

    # Build dataloader from paths
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

    loss_list = []
    result_list = []
    with torch.no_grad():
        for _, val_data in tqdm(
            enumerate(loader), total=len(loader), desc="Predicting age..."
        ):
            images = val_data["vol_data"].float().to(args.device)
            labels = val_data["vol_data"].meta["age"][None].to(args.device)
            outputs = model(images)["pred_out"]
            val_loss = torch.nn.L1Loss()(outputs, labels.float())

            loss_list.append(val_loss.item())
            result_dict = {
                "loss": val_loss.item(),
                "label": labels.cpu().numpy().item(),
                "pred": outputs.cpu().numpy().item(),
            }
            result_list.append(result_dict)

        avg_val_loss = sum(loss_list) / len(loss_list)

    return avg_val_loss, result_list
