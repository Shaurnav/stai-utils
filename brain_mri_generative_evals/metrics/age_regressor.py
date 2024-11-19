import torch

from brain_mri_generative_evals.models.unet3d.model import UNet3D
from brain_mri_generative_evals.models.finetune_model import FinetuneModel
from brain_mri_generative_evals.util import create_dataloader


def evaluate_age_regression(directory_path, args):
    val_loader = create_dataloader(directory_path)
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
    checkpoint_path = (
        "/hai/scratch/alanqw/models/age_regressor/epoch148_trained_model.pth.tar"
    )
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    labels_list_val = []
    outputs_list_val = []
    with torch.no_grad():
        val_losses = []
        for val_data in val_loader:
            images, labels = val_data["image"].float().to(args.device), val_data[
                "age"
            ].float().to(args.device)
            outputs = model(images)["pred_out"]
            print(f"val_labels: {labels.shape}")
            print(f"val_outputs: {outputs.shape}")
            val_loss = torch.nn.L1Loss()(outputs, labels.float())
            val_losses.append(val_loss.item())
            labels_list_val.extend(labels.cpu().numpy())
            outputs_list_val.extend(outputs.cpu().numpy())

        avg_val_loss = sum(val_losses) / len(val_losses)

    return avg_val_loss
