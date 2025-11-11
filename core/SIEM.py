import torch
import torch.nn.functional as F
import clip
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from skimage import color
from scipy.ndimage import gaussian_filter

def Degradation_suppression(image, micro=1.0, delta=0.7, mask_threshold=0.85, mask_sigma=50):
    """
    Parameters:
        image (numpy.ndarray): Input underwater image (RGB format).
        micro (float): Compensation parameter for the 'a' channel.
        delta (float): Compensation parameter for the 'b' channel.
        mask_threshold (float): Threshold for generating the mask.

    Returns:
        numpy.ndarray.
    """
    # Convert RGB image to Lab color space
    lab_image = color.rgb2lab(image / 255.0)
    L, a, b = lab_image[..., 0], lab_image[..., 1], lab_image[..., 2]

    # Calculate the mean of the RGB channels (r, g, b)
    mean_rgb = np.mean(image, axis=-1) / 255.0

    # Generate mask M
    mask_threshold = np.percentile(mean_rgb, 70)
    mask = np.ones_like(mean_rgb)
    mask[mean_rgb > mask_threshold] = 0
    mask = gaussian_filter(mask, sigma=mask_sigma)

    # Apply Gaussian blur
    G_a = gaussian_filter(a, sigma=mask_sigma)
    G_b = gaussian_filter(b, sigma=mask_sigma)

    # Compensate 'a' and 'b' channels
    a_compensated = a - micro * mask * G_a
    b_compensated = b - delta * mask * G_b

    # Merge channels back to Lab image
    compensated_lab = np.stack([L, a_compensated, b_compensated], axis=-1)

    # Convert Lab image back to RGB
    compensated_rgb = color.lab2rgb(compensated_lab)

    # Scale to 0-255 and convert to uint8
    compensated_rgb = np.clip(compensated_rgb * 255, 0, 255).astype(np.uint8)

    return compensated_rgb

def process_batch(batch_ram, ram_model, clip_model, device):

    batch_prompts = []

    for i in range(batch_ram.size(0)):

        single_image = batch_ram[i]
        img_np = single_image.detach().cpu()
        img_np = img_np.permute(1, 2, 0).numpy()
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        ref = Degradation_suppression(img_np)
        ref = Image.fromarray(ref)

        single_image = tensor_transforms(ref).unsqueeze(0).to(device)
        single_image = ram_transforms(single_image)
        single_image = single_image.half()
        with torch.no_grad():
            tags, _ = ram_model.generate_tag(single_image)
        validation_prompt = f"{tags[0]}"
        batch_prompts.append(validation_prompt)
    clip_model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(batch_prompts).to(device)
        text_features = clip_model.encode_text(tokens)

    return text_features
