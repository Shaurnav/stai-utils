import torch
from generative.metrics import MultiScaleSSIMMetric
from brain_mri_generative_evals.util import create_dataloader


def compute_pairwise_msssim(img_directory, N=1000):
    tot_metric = 0
    count = 0
    msssim = MultiScaleSSIMMetric(spatial_dims=3, kernel_size=9)
    loader1 = create_dataloader(img_directory)
    loader2 = create_dataloader(img_directory)
    for i, data1 in enumerate(loader1):
        for j, data2 in enumerate(loader2):
            if count >= N:
                break
            if i != j:
                img1 = torch.tensor(data1["image"]).float()
                img2 = torch.tensor(data2["image"]).float()
                tot_metric += msssim(img1, img2).item()
                count += 1

    return tot_metric / count
