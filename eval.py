import sys

import argparse
import numpy as np
import os
import torch
from monai.utils import first, set_determinism
import json
import pandas as pd
import sys

from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from brain_mri_generative_evals.metrics.sex_classifier import (
    evaluate_sex_classification,
)
from brain_mri_generative_evals.metrics.age_regressor import evaluate_age_regression
from brain_mri_generative_evals.metrics.fid import (
    evaluate_fid_imagenet2d,
    evaluate_fid_medicalnet3d,
)
from brain_mri_generative_evals.metrics.msssim import compute_pairwise_msssim
from brain_mri_generative_evals.sample.generate import (
    generate_real_samples_to_dir,
    generate_synthetic_samples_to_dir,
)
from utils import define_instance
from util.dataset_utils import prepare_dataloader_from_list


def load_saved_brain_image():
    return np.load("images_moving.npy")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate multiple model checkpoints")
    parser.add_argument(
        "-e",
        "--environment-file",
        type=str,
        required=True,
        help="Path to the environment configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--evaluation_output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Resume latest checkpoint available",
    )
    args = parser.parse_args()

    # Load model configuration
    with open(args.config_file, "r") as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)
    with open(args.environment_file, "r") as f:
        environment_config = json.load(f)

    for k, v in environment_config.items():
        setattr(args, k, v)
    return args


def get_data(args):
    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    size_divisible = 2 ** (len(args.diffusion_def["num_channels"]) - 1)
    if args.dataset_type == "brain_tumor":
        train_loader, val_loader = prepare_dataloader(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
        )

    elif args.dataset_type == "hcp_ya_T1":

        train_loader, val_loader = prepare_dataloader_extract_dataset_custom(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            # base_dir= base_dir,
            all_files=all_files,
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
        )
    elif args.dataset_type == "T1_all":
        train_loader, val_loader = prepare_dataloader_from_list(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
            cross_attention_dim=args.diffusion_def["cross_attention_dim"],
            expand_token_times=args.diffusion_train["expand_token_times"],
        )

    else:
        raise ValueError(f"Unsupported dataset type specified: {args.dataset_type}")

    return {
        "train": train_loader,
        "val": val_loader,
    }


def generate_conditions(total_length):
    # Generate ages from 0 to 80, linearly spaced to match the total length
    ages = np.linspace(10, 80, total_length)

    # Generate half 0's and half 1's for sex
    sexes = np.array([0, 1] * (total_length // 2 + 1))[:total_length]

    # Combine into a list of lists
    data = [[age, sex] for age, sex in zip(ages, sexes)]

    return data


def main(args):
    set_determinism(seed=42)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories
    args.model_dir = os.path.join(args.base_model_dir, args.run_name)
    args.evaluation_output_dir = os.path.join(args.model_dir, "evaluation")
    if not os.path.exists(args.evaluation_output_dir):
        os.makedirs(args.evaluation_output_dir)
    args.real_samples_dir = os.path.join(args.evaluation_output_dir, "real_samples")
    args.real_medicalnet_feat_dir = os.path.join(
        args.real_samples_dir, "medicalnet_features"
    )
    args.real_imagenet_feat_dir = os.path.join(
        args.real_samples_dir, "imagenet_features"
    )
    args.syn_samples_dir = os.path.join(args.evaluation_output_dir, "syn_samples")
    args.syn_medicalnet_feat_dir = os.path.join(
        args.syn_samples_dir, "medicalnet_features"
    )
    args.syn_imagenet_feat_dir = os.path.join(args.syn_samples_dir, "imagenet_features")
    if not os.path.exists(args.real_samples_dir):
        os.makedirs(args.real_samples_dir)
    if not os.path.exists(args.syn_samples_dir):
        os.makedirs(args.syn_samples_dir)
    if not os.path.exists(args.real_medicalnet_feat_dir):
        os.makedirs(args.real_medicalnet_feat_dir)
    if not os.path.exists(args.real_imagenet_feat_dir):
        os.makedirs(args.real_imagenet_feat_dir)
    if not os.path.exists(args.syn_samples_dir):
        os.makedirs(args.syn_samples_dir)
    if not os.path.exists(args.syn_medicalnet_feat_dir):
        os.makedirs(args.syn_medicalnet_feat_dir)
    if not os.path.exists(args.syn_imagenet_feat_dir):
        os.makedirs(args.syn_imagenet_feat_dir)
    csv_filename = "results.csv"
    csv_filepath = os.path.join(args.evaluation_output_dir, csv_filename)
    print("csv_filepath", csv_filepath)

    args.autoencoder_dir = os.path.join(args.model_dir, "autoencoder")
    args.diffusion_dir = os.path.join(args.model_dir, "diffuion")
    autoencoder_checkpoint_path = os.path.join(
        args.autoencoder_dir, "autoencoder_best.pt"
    )
    unet_checkpoint_path = os.path.join(args.diffusion_dir, "diffusion_unet_best.pt")
    unet_checkpoint = torch.load(unet_checkpoint_path, map_location=args.device)
    autoencoder_checkpoint = torch.load(
        autoencoder_checkpoint_path, map_location=args.device
    )

    train_loader = get_data(args)["train"]

    autoencoder = define_instance(args, "autoencoder_def").to(args.device)
    if "state_dict" in autoencoder_checkpoint:
        autoencoder.load_state_dict(autoencoder_checkpoint["state_dict"])
    else:
        autoencoder.load_state_dict(autoencoder_checkpoint)
    unet = define_instance(args, "diffusion_def").to(args.device)
    if "state_dict" in unet_checkpoint:
        unet.load_state_dict(unet_checkpoint["state_dict"])
    else:
        unet.load_state_dict(unet_checkpoint)
    cond_linear = torch.nn.Linear(2, 256).to(args.device)
    cond_linear.load_state_dict(unet_checkpoint["condition_embedder"])

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )
    # scheduler = DDIMScheduler(
    #     num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
    #     # schedule="scaled_linear_beta",
    #     # beta_start=args.NoiseScheduler["beta_start"],
    #     # beta_end=args.NoiseScheduler["beta_end"],
    # )

    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

    check_data = first(train_loader)
    with torch.no_grad():
        z = autoencoder.encode_stage_2_inputs(
            check_data["image"].to(args.device).float()
        )
        args.latent_shape = z.shape[-4:]
        print(args.latent_shape)

    print("Generating synthetic samples....")
    if (
        args.autoencoder_def["_target_"]
        == "generative.networks.nets.AutoencoderKLRegistration"
    ):
        image_moving = next(iter(train_loader))["image"].to(args.device).float()
    if (
        args.autoencoder_def["_target_"]
        == "generative.networks.nets.AutoencoderKLTemplateRegistration"
    ):
        image_moving = load_saved_brain_image()
        image_moving = torch.tensor(image_moving).float().to(args.device)
    conditions = generate_conditions(args.num_samples)
    print(conditions)
    scheduler.set_timesteps(1000)
    generate_synthetic_samples_to_dir(
        args.syn_samples_dir,
        unet,
        autoencoder,
        scheduler,
        inferer,
        conditions,
        args,
        img_template=image_moving,
        cond_linear=cond_linear,
    )

    print("Generating real samples....")
    generate_real_samples_to_dir(args.real_samples_dir, train_loader, args)

    sex_acc_real = evaluate_sex_classification(args.real_samples_dir, args)
    sex_acc_syn = evaluate_sex_classification(args.syn_samples_dir, args)

    age_mae_real = evaluate_age_regression(args.real_samples_dir, args)
    age_mae_syn = evaluate_age_regression(args.syn_samples_dir, args)

    fid_2d_score = evaluate_fid_imagenet2d(
        args.real_samples_dir,
        args.syn_samples_dir,
        args.real_imagenet_feat_dir,
        args.syn_imagenet_feat_dir,
        args.device,
        args.skip_existing,
    )
    fid_3d_score = evaluate_fid_medicalnet3d(
        args.real_samples_dir,
        args.syn_samples_dir,
        args.real_medicalnet_feat_dir,
        args.syn_medicalnet_feat_dir,
        args.device,
        args.skip_existing,
    )

    ms_ssim_score = compute_pairwise_msssim(args.real_samples_dir, N=10)

    evaluation_results = {
        "age_mae_real": age_mae_real,
        "age_mae_synthetic": age_mae_syn,
        "sex_accuracy_real": sex_acc_real,
        "sex_accuracy_synthetic": sex_acc_syn,
        "fid_2d": fid_2d_score,
        "fid_3d": fid_3d_score,
        "ms_ssim_generated": ms_ssim_score,
    }

    new_df = pd.DataFrame([evaluation_results])

    # Adjust for the case when an item is a dict
    for column in new_df.columns:
        if isinstance(new_df[column].iloc[0], dict):
            new_df[column] = new_df[column].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

    if os.path.exists(csv_filepath):
        # If file exists, read existing data and append new row
        existing_df = pd.read_csv(csv_filepath)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If file doesn't exist, create new file with the data
        updated_df = new_df
    print("csv_filepath", csv_filepath)
    updated_df.to_csv(csv_filepath, index=False)

    print(f"Evaluation complete. Results saved in {args.evaluation_output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
