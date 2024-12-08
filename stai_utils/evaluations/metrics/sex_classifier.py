import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from stai_utils.evaluations.models.unet3d.model import UNet3D
from stai_utils.evaluations.models.finetune_model import FinetuneModel
from stai_utils.evaluations.models.resnet import resnet50


def load_model(checkpoint_path, device):
    model = resnet50(shortcut_type="B").to(device)
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(2048, 1)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_sex_classification(loader, args):
    model_checkpoint = "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/condition_predictor/condition_predictor_ckpt/20241117_174033/step_140000.pth"
    model = load_model(model_checkpoint, args.device)

    correct_sex = 0
    total_sex = 0
    result_list = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Predicting sex... {i}/{len(loader)}")
            images = batch["image"].to(args.device).float()
            sex_labels = batch["sex"].to(args.device)
            outputs = model(images)
            sex_preds = (torch.sigmoid(outputs) > 0.5).long()
            print("sex_preds: ", sex_preds.shape)
            print("sex_labels: ", sex_labels.shape)

            correct_sex += (sex_preds == sex_labels).sum().item()
            total_sex += sex_labels.size(0)

            result_dict = {
                "acc": (sex_preds == sex_labels).sum().item(),
                "pred": sex_preds.cpu().numpy().item(),
                "label": sex_labels.cpu().numpy().item(),
            }
            result_list.append(result_dict)

    accuracy = correct_sex / total_sex
    print(f"Validation Accuracy: {accuracy:.4f}")

    # # Calculate additional metrics
    # true_positives = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    # true_negatives = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    # false_positives = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    # false_negatives = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))

    # sensitivity = (
    #     true_positives / (true_positives + false_negatives)
    #     if (true_positives + false_negatives) > 0
    #     else 0
    # )
    # specificity = (
    #     true_negatives / (true_negatives + false_positives)
    #     if (true_negatives + false_positives) > 0
    #     else 0
    # )

    # print(f"Sensitivity: {sensitivity:.4f}")
    # print(f"Specificity: {specificity:.4f}")
    return accuracy, result_list  # , sensitivity, specificity


def evaluate_sex_classification_unet3dencoder(loader, args):
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

    labels_list_val, outputs_list_val = evaluate_model(loader, model, args.device)
    accuracy = accuracy_score(labels_list_val, outputs_list_val)
    return accuracy
