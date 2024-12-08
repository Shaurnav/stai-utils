import torch

from stai_utils.evaluations.models.unet3d.model import UNet3D
from stai_utils.evaluations.models.finetune_model import FinetuneModel


def get_model():
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
    return model


def evaluate_age_regression(loader, args):
    model = get_model()
    model.to(args.device)
    model.eval()

    loss_list = []
    result_list = []
    with torch.no_grad():
        for val_data in loader:
            images = val_data["image"].float().to(args.device)
            labels = val_data["age"][None].float().to(args.device)
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
