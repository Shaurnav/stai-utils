import torch
from sklearn.metrics import accuracy_score

from brain_mri_generative_evals.models.unet3d.model import UNet3D
from brain_mri_generative_evals.models.finetune_model import FinetuneModel
from brain_mri_generative_evals.util import create_dataloader


def evaluate_sex_classification(directory_path, args):
    def evaluate_model(val_loader, model, device):
        model.eval()
        labels_list_val = []
        outputs_list_val = []
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].float().to(device), val_data[
                    "sex"
                ].float().to(device)
                print(f"val_images shape: {val_images.shape}")
                val_outputs = model(val_images)["pred_out"]
                print(f"sex val_labels: {val_labels.shape}")
                print(f"sex val_outputs: {val_outputs.shape}")
                labels_list_val.extend(val_labels.cpu().numpy())
                # Convert the model output to class predictions
                predicted_classes = torch.argmax(val_outputs, dim=1).cpu().numpy()
                outputs_list_val.extend(predicted_classes)
        return labels_list_val, outputs_list_val

    val_loader = create_dataloader(directory_path)
    print(f"Number of validation images: {len(val_loader.dataset)}")

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
        output_dim=2,
        dim=3,
        use_checkpoint=False,
    )
    checkpoint_path = "/hai/scratch/alanqw/models/sex_classifier/sex_classifier_epoch129_trained_model.pth.tar"
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    labels_list_val, outputs_list_val = evaluate_model(val_loader, model, args.device)
    accuracy = accuracy_score(labels_list_val, outputs_list_val)
    return accuracy
