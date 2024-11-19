import argparse
import json

import os
import sys


import torch
import torch.nn as nn

import torch.amp
import wandb
import nibabel as nib


sys.path.append("/hai/scratch/fangruih/monai-tutorials/3d_regression/")
sys.path.append("/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/")
from resnet3D import resnet50

from util.dataset_utils import prepare_dataloader_from_list, prepare_val_dataloader_from_directory
from utils import setup_ddp

def load_model(checkpoint_path, device):
    model = resnet50(shortcut_type='B').to(device)
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 1)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer_on_directory(model, directory, device):
    print(f"Inferring on directory: {directory}")
    correct_sex = 0
    total_sex = 0
    for file_name in os.listdir(directory):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(directory, file_name)
            image = nib.load(file_path).get_fdata()
            image = torch.tensor(image, dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                sex_pred = torch.sigmoid(output) > 0.5
                print(f"File: {file_name}, Predicted Sex: {'Male' if sex_pred else 'Female'}")
                total_sex += 1
                # Assuming ground truth is encoded in filename
                true_sex = 'male' in file_name.lower()
                correct_sex += (sex_pred.item() == true_sex)
    
    if total_sex > 0:
        accuracy = correct_sex / total_sex
        print(f"Directory Inference Accuracy: {accuracy:.4f}")

def infer_on_validation_set(model, val_loader, device):
    print("Inferring on validation set")
    correct_sex = 0
    total_sex = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            
            images = batch["image"].to(device)
            
            sex_labels = batch["sex"].to(device)
            outputs = model(images)
            sex_preds = torch.sigmoid(outputs) > 0.5
            correct_sex += (sex_preds == sex_labels).sum().item()
            total_sex += sex_labels.size(0)
            all_preds.extend(sex_preds.cpu().numpy())
            all_labels.extend(sex_labels.cpu().numpy())
            # Print predictions and labels for each sample in batch
            for pred, label in zip(sex_preds.cpu().numpy(), sex_labels.cpu().numpy()):
                print(f"Prediction: {'Male' if pred else 'Female'}, "
                      f"True Label: {'Male' if label else 'Female'}, "
                      f"Match: {pred == label}")
    
    accuracy = correct_sex / total_sex
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Calculate additional metrics
    true_positives = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
    
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    return accuracy, sensitivity, specificity



if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Inference Predictor")
    parser.add_argument("-m", "--model-checkpoint", default="/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/condition_predictor/condition_predictor_ckpt/20241117_174033/step_140000.pth", help="Path to the model checkpoint")
    parser.add_argument("-d", "--validation_dir", help="Directory containing image files for inference")
    parser.add_argument("-v", "--validation", action='store_true', help="Use validation dataset for inference")
    parser.add_argument("-e", "--environment-file", default="../config/environment_config/environment_t1_all.json", help="Environment JSON file")
    parser.add_argument("-c", "--config-file", default="../config/diffusion/config_diffusion_condition_vae_48g.json", help="Config JSON file")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_checkpoint, device)

    # build a validation loader 
    val_loader = None
    if args.validation:
        # Assuming you have a function to prepare the validation dataloader
        size_divisible = 2 ** (len(args.diffusion_def["num_channels"]) - 1)
        ddp_bool = args.gpus > 1  # whether to use distributed data parallel
        if ddp_bool:
            rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            dist, device = setup_ddp(rank, world_size)
        else:
            rank = 0
            world_size = 1
            device = 0

        _, val_loader, _, _ =  prepare_dataloader_from_list(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            # rank=rank,
            # world_size=world_size,
            # cache=1.0,
            # size_divisible=size_divisible,
            # amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
            cross_attention_dim=args.diffusion_def["cross_attention_dim"],
            expand_token_times= args.diffusion_train["expand_token_times"],
        )
    else:
        val_loader = prepare_val_dataloader_from_directory(
            args.validation_dir,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            cross_attention_dim=args.diffusion_def["cross_attention_dim"],
            expand_token_times= args.diffusion_train["expand_token_times"],
        )
    
    accuracy, sensitivity, specificity = infer_on_validation_set(model, val_loader, device)
    print(f"Sex Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
