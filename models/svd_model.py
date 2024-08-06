import torch

def load_svd_model(unet, device, diffusion_model_learning_rate):
    optim_params = []
    optim_params_1d = []
    for n, p in unet.named_parameters():
        if "delta" in n:
            p.requires_grad = True
            if "norm" in n:
                optim_params_1d.append(p)
            else:
                optim_params.append(p)
    
    optimizer = torch.optim.AdamW(
        [{"params": optim_params}, {"params": optim_params_1d, "lr": 1e-6}],
        lr=diffusion_model_learning_rate,
    )
    return unet, optimizer
