import os
import torch
import numpy as np


def generate_synthetic_samples_to_dir(
    output_dir,
    unet,
    autoencoder,
    scheduler,
    inferer,
    conditions,
    args,
    img_template=None,
    cond_linear=None,
):

    latent_shape = args.latent_shape

    for i, condition_list in enumerate(conditions):
        filename = os.path.join(output_dir, f"synimg_{i}.npz")
        save_path = os.path.join(output_dir, filename)
        if args.skip_existing and os.path.exists(save_path):
            print(f"Skipping because {save_path} already exists...")
            continue

        noise = torch.randn((1, *latent_shape))
        noise = noise.to(args.device)

        condition = torch.tensor(condition_list)[None].float().to(args.device)
        cond_embedding = cond_linear(condition).unsqueeze(1)
        synthetic_images = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
            img_template=img_template,
            conditioning=cond_embedding,
        )

        np.savez(
            save_path,
            image=synthetic_images[0].cpu().detach().numpy(),
            age=condition_list[0],
            sex=condition_list[1],
        )


def generate_real_samples_to_dir(output_dir, loader, args):
    for i in range(len(loader)):
        filename = os.path.join(output_dir, f"realimg_{i}.npz")
        save_path = os.path.join(output_dir, filename)
        if args.skip_existing and os.path.exists(save_path):
            print(f"Skipping because {save_path} already exists...")
            continue
        if i >= args.num_samples:
            break
        data = loader.dataset[i]
        image = data["image"]
        age = data["age"]
        sex = data["sex"]

        np.savez(save_path, image=image[0], age=age, sex=sex)
