import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import requests
from io import BytesIO
import random

def load_im(im_path, target_size=224):
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
        
    tforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])

    im = tforms(im)
    return 2.*im - 1.

def load_cond_im(im_path, device):
    im = Image.open(im_path).convert("RGB")
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).to(device).unsqueeze(0)
    return inp

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def flip_transforms(im):
    horizontal_flip = transforms.functional.hflip(im)
    vertical_flip = transforms.functional.vflip(im)
    hv_flip = transforms.functional.vflip(horizontal_flip)  
    return [horizontal_flip, vertical_flip, hv_flip]

def horizontal_flip(im):
    horizontal_flip = transforms.functional.hflip(im)
    return [horizontal_flip]

def image_augmentations(im_path, augmentations, target_size=512):
    im = Image.open(im_path).convert("RGB")

    img_base_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    cond_base_transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])

    aug_transforms = {
        "rotate": transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(round(target_size*1.3)),
                                      transforms.CenterCrop(target_size)
                                     ]),
        "flip": flip_transforms,
        "h_flip": horizontal_flip,
        "crop": transforms.RandomResizedCrop(size=target_size, scale=(0.6, 0.6), ratio=(1.0, 1.0)),
    }

    augmented_images = []
    
    # for the base image
    img_base = img_base_transform(im)
    img = 2. * img_base - 1.
    cond = cond_base_transform(img_base)
    augmented_images.append((img, cond))

    # for the augmented images
    for aug in augmentations:
        if aug == "flip" or aug == "h_flip":
            flipped_images = aug_transforms[aug](im)
            for flipped_im in flipped_images:
                img_aug = img_base_transform(flipped_im)
                img = 2. * img_aug - 1.
                cond = cond_base_transform(img_aug)
                augmented_images.append((img, cond))
        elif aug == "flip_1":
            flipped_images = aug_transforms["flip"](im)
            # random choice
            flipped_image = random.choice(flipped_images)
            img_aug = img_base_transform(flipped_image)
            img = 2. * img_aug - 1.
            cond = cond_base_transform(img_aug)
            augmented_images.append((img, cond))
        elif aug == "none":
            continue
        else:
            aug_transform = transforms.Compose([
                aug_transforms[aug],
                img_base_transform,  
            ])
            img_aug = aug_transform(im)
            img = 2. * img_aug - 1.
            cond = cond_base_transform(img_aug)
            augmented_images.append((img, cond))

    return augmented_images