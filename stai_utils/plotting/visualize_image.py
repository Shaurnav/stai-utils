# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from monai.utils.type_conversion import convert_to_numpy

import matplotlib.pyplot as plt
import torch
import io
from PIL import Image as PILImage


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img


def visualize_one_slice_in_3d_image_greyscale(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    # draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    # array[..., np.newaxis]
    return draw_img[..., np.newaxis]


def plot_latent(z, title=None):
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))  # 2 rows, 4 columns

    # Iterate through the 8 channels
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(z[0, i, 20], cmap="gray")  # Display the channel
        ax.set_title(f"Channel {i+1}")
        ax.axis("off")  # Hide axes for a cleaner look

        # Add colorbar next to each subplot
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04
        )  # Adjust size & spacing

    if title:
        plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def figure_to_tensor(fig):
    """
    Convert a matplotlib figure to a PyTorch tensor.

    Args:
        fig: matplotlib figure object

    Returns:
        torch.Tensor of shape (3, H, W) with values in range [0, 1]
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    # Convert to PIL Image
    img = PILImage.open(buf).convert("RGB")

    # Convert to tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Clean up
    plt.close(fig)

    return img_tensor


def visualize_displacement_field(
    disp,
    batch_idx: int = 0,
    figsize: tuple[float, float] = (15, 5),
    show_quiver: bool = False,
    quiver_stride: int = 8,
):
    """
    Visualize mid‐slice magnitude (and optionally quiver arrows) of a 3D displacement field.

    Args:
        disp: displacement field, shape (B, 3, D, H, W) or (3, D, H, W)
        batch_idx: which batch element to visualize (if B>1)
        figsize: matplotlib figure size
        show_quiver: if True, overlay quiver arrows on magnitude
        quiver_stride: downsampling factor for quiver arrows
    """
    # to numpy
    if isinstance(disp, torch.Tensor):
        disp = disp.detach().cpu().numpy()
    # squeeze batch
    if disp.ndim == 5:
        disp = disp[batch_idx]

    # disp now shape (3, D, H, W)
    # compute magnitude
    mag = np.sqrt(np.sum(disp**2, axis=0))
    D, H, W = mag.shape

    # choose mid‐slice indices
    z0, y0, x0 = D // 2, H // 2, W // 2

    slices = {
        f"Axial (Z={z0})": mag[z0, :, :],
        f"Coronal (Y={y0})": mag[:, y0, :],
        f"Sagittal (X={x0})": mag[:, :, x0],
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (title, im) in zip(axes, slices.items()):
        ax.imshow(im.T, origin="lower")
        ax.set_title(title)
        ax.axis("off")

        if show_quiver:
            # get corresponding vector‐components for arrows
            if "Axial" in title:
                u = disp[0, z0, ::quiver_stride, ::quiver_stride]
                v = disp[1, z0, ::quiver_stride, ::quiver_stride]
            elif "Coronal" in title:
                u = disp[0, ::quiver_stride, y0, ::quiver_stride]
                v = disp[2, ::quiver_stride, y0, ::quiver_stride]
            else:  # Sagittal
                u = disp[1, ::quiver_stride, ::quiver_stride, x0]
                v = disp[2, ::quiver_stride, ::quiver_stride, x0]

            X, Y = np.meshgrid(
                np.arange(u.shape[1]) * quiver_stride,
                np.arange(u.shape[0]) * quiver_stride,
            )
            ax.quiver(
                X, Y, u.T, v.T, angles="xy", scale_units="xy", scale=1.0, width=0.002
            )

    plt.tight_layout()
    return figure_to_tensor(fig)
