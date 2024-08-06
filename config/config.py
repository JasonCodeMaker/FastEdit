import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Stable Diffusion model.")
    parser.add_argument("--dataset", type=str, default='tedbench/input_list.json', help="Training Dataset to use.")
    parser.add_argument("--index", type=int, default=1, help="Current index of the image.")
    parser.add_argument("--method", type=str, default='lora', help="Finetune method")
    parser.add_argument("--aug", type=str, default=None, help="Image Augmentation Method")
    parser.add_argument("--outdir", type=str, default="output_v4.4", help="Output directory")
    parser.add_argument("--timestep", type=str, default="fixed", help="Timestep value for the model")
    parser.add_argument("--seed", type=int, default=1, help="seed for the model")

    # parser.add_argument("--dataset", type=str, required=True, help="Training Dataset to use.")
    # parser.add_argument("--index", type=int, required=True, help="Current index of the image.")
    # parser.add_argument("--method", type=str, required=True, help="Finetune method")
    # parser.add_argument("--aug", type=str, required=True, help="Image Augmentation Method")
    # parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    # parser.add_argument("--timestep", type=str, default="fixed", help="Timestep value for the model")
    # parser.add_argument("--seed", type=int, default=1, help="seed for the model")
    return parser.parse_args()
