from peft import LoraConfig, LoraModel 
import torch

def load_lora_model(unet, device, diffusion_model_learning_rate):
 
    for param in unet.parameters():
        param.requires_grad_(False)
    
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=diffusion_model_learning_rate,
    )
    return unet, optimizer
