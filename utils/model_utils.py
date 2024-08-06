import torch
from utils.image_utils import numpy_to_pil
from tqdm.auto import tqdm

from transformers import CLIPModel, CLIPProcessor
from models.lora_model import load_lora_model
from models.svd_model import load_svd_model
from diffusers import StableDiffusionImageVariationPipeline

@torch.no_grad()
def sample_image(image_embeddings, device, unet, scheduler, guidance_scale=1.0, seed=0, start_code=None):

    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        target_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    else:
        target_embeddings = image_embeddings


    # generate initial latents noise
    if start_code is not None:
        latents = start_code
    else:
        latents_shape = (1, unet.config.in_channels, 512 // 8, 512 // 8)
        latents_dtype = target_embeddings.dtype
        if seed != -1:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=latents_dtype)

    # set timesteps
    scheduler.set_timesteps(50)
    timesteps_tensor = scheduler.timesteps.to(device)

    # generate images
    for t in tqdm(timesteps_tensor, desc="Sampling"):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=target_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents

def decode_image(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    samples = vae.decode(latents, return_dict=False)[0]
    ims = (samples / 2 + 0.5).clamp(0, 1)
    x_sample = ims.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(x_sample)[0]

    return image

def encode_image(image, device, image_encoder, feature_extractor):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    return image_embeddings

def configure_model(method, device, prompt):
    # Set model parameters based on the method
    model_params = {
        "lora": {"lr": 4e-4, "steps": 50}, #1e-4
        "baseline": {"lr": 1e-5, "steps": 300},
        "SVD": {"lr": 1e-3, "steps": 300}
    }
    lr = model_params[method]["lr"]
    optimization_steps = model_params[method]["steps"]
    
    # Load Stable Diffusion model
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", revision="v2.0").to(device)
    vae, scheduler, image_encoder, feature_extractor = sd_pipe.vae, sd_pipe.scheduler, sd_pipe.image_encoder, sd_pipe.feature_extractor

    # Freeze components
    for component in [vae, image_encoder]:
        component.requires_grad_(False).to(device).eval()

    # Load specific UNet configuration
    if method == "SVD":
        from svdiff_pytorch import load_unet_for_svdiff
        unet = load_unet_for_svdiff("models/unet/").to(device)
    else:
        unet = sd_pipe.unet
        unet.requires_grad_(False).to(device).eval()

    # Configure optimizer
    optimizer = None
    if method in ["lora", "SVD"]:
        unet, optimizer = load_lora_model(unet, device, lr) if method == "lora" \
            else load_svd_model(unet, device, lr)
    else:  # baseline method
        optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # Load text encoder and process prompt
    text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    text_embeddings = text_encoder.get_text_features(**text_processor(prompt, return_tensors="pt")).unsqueeze(0).to(device)

    return unet, optimizer, optimization_steps, vae, scheduler, image_encoder, feature_extractor, text_embeddings


def mixup_embeddings(image_embeddings, text_embeddings, mixup_alpha=1.0):
    mixup_weights = torch.distributions.beta.Beta(mixup_alpha, mixup_alpha).sample((image_embeddings.shape[0], 1)).to(image_embeddings.device).detach()
    mixed_embeddings = mixup_weights * image_embeddings + (1 - mixup_weights) * text_embeddings

    return mixed_embeddings, mixup_weights


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
