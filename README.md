# FastEdit: Fast Text Guided Single Image Editing via Semantic-Aware Diffusion Fine-Tuning

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://fastedit-sd.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2408.03355)

FastEdit is a fast text-guided single-image editing method with semantic-aware diffusion fine-tuning, accelerating the editing process to only 17 seconds.

## Authors

- Zhi Chen*
- Zecheng Zhao*
- Yadan Luo
- Zi Huang

*The University of Queensland*

## Setup

To set up the project environment, run the following commands:

```bash
# Install PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
pip install diffusers==0.27.2
pip install transformers==4.25.1
pip install accelerate==0.25.0
```

## Citation
```bibtex
@article{chen2024fastedit,
    title={FastEdit: Fast Text-Guided Single-Image Editing via Semantic-Aware Diffusion Fine-Tuning},
    author={Chen, Zhi and Zhao, Zecheng and Luo, Yadan and Huang, Zi},
    journal={arXiv preprint arXiv:2408.03355},
    year={2024}
}
```
