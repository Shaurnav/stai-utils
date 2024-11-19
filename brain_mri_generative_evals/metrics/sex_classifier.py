import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from brain_mri_generative_evals.models.unet3d.model import UNet3D
from brain_mri_generative_evals.models.finetune_model import FinetuneModel
from brain_mri_generative_evals.util import create_dataloader
from brain_mri_generative_evals.models.resnet import resnet50


def load_model(checkpoint_path, device):
    model = resnet50(shortcut_type="B").to(device)
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(2048, 1)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_sex_classification(directory_path, args):
    model_checkpoint = "/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/condition_predictor/condition_predictor_ckpt/20241117_174033/step_140000.pth"
    model = load_model(model_checkpoint, args.device)

    # Load validation data
    val_loader = create_dataloader(directory_path)

    correct_sex = 0
    total_sex = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:

            images = batch["image"].to(args.device).float()
            print(images.shape)
            sex_labels = batch["sex"].to(args.device)
            outputs = model(images)
            sex_preds = (torch.sigmoid(outputs) > 0.5).long()
            print("sex_preds: ", sex_preds.shape)
            print("sex_labels: ", sex_labels.shape)

            correct_sex += (sex_preds == sex_labels).sum().item()
            total_sex += sex_labels.size(0)
            all_preds.extend(sex_preds.cpu().numpy())
            all_labels.extend(sex_labels.cpu().numpy())
            # # Print predictions and labels for each sample in batch
            # for pred, label in zip(sex_preds.cpu().numpy(), sex_labels.cpu().numpy()):
            #     print(f"Prediction: {'Male' if pred else 'Female'}, "
            #           f"True Label: {'Male' if label else 'Female'}, "
            #           f"Match: {pred == label}")
            print("correct_sex: ", correct_sex)
            print("total_sex: ", total_sex)

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
    return accuracy  # , sensitivity, specificity


def evaluate_sex_classification_unet3dencoder(directory_path, args):
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
