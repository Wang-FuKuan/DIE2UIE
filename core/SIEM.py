import torch
import torch.nn.functional as F
import clip
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
from core.ram.models.ram_lora import ram
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

def process_batch(batch_ram, clip_model, device):

    model = clip_model
    RAM = ram(pretrained='/models/ram_swin_large_14m.pth',
                pretrained_condition = None,
                image_size=384,
                vit='swin_l')
    RAM.eval().to(device, dtype=torch.float16)

    tensor_transforms = transforms.Compose([
                    transforms.ToTensor(),
                ])

    ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    ram_model = RAM
    batch_prompts = []

    for i in range(batch_ram.size(0)):

        single_image = batch_ram[i]
        ref = Degradation_suppression(np.array(single_image))
        ref = Image.fromarray(ref)
        to_pil = ToPILImage()
        single_image = to_pil(ref)

        single_image = tensor_transforms(single_image).unsqueeze(0).to(device)
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