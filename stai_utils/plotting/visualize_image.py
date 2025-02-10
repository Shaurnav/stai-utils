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
