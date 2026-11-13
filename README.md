# VS Code, the UT HPC cluster, and a look inside a Vision Transformer

Tutorial material for **GenAI Models and Robotic Applications (2026-1B)**, University of Twente.

**Start here: [GUIDE.md](GUIDE.md)**, the step-by-step guide for Windows and Ubuntu.

## What you will do

1. Install VS Code, Git, Python, and eduVPN.
2. Connect VS Code to the UT HPC cluster over SSH.
3. Clone this repository, create a virtual environment, and install PyTorch on the cluster.
4. Submit a small GPU job that prints the PyTorch version, the GPU name, and its memory.
5. Do the same on your laptop and run a small Vision Transformer that shows how an image becomes
   patches, embeddings, attention, and a prediction.
6. (Optional) Add GitHub Copilot to VS Code.

## Repository contents

```
vscode-hpc-tutorial/
├── GUIDE.md               step-by-step guide
├── requirements.txt       packages for the laptop demo
├── hpc/
│   ├── check_gpu.py       prints machine, PyTorch, GPU name, VRAM, and a speed test
│   └── check_gpu.sbatch   Slurm job script (one GPU, at most 5 minutes)
└── demo/
    ├── vit_demo.py        the Vision Transformer demo
    └── images/            sample images (see CREDITS.md)
```

## Quick start (if you already know the tools)

On the cluster:

```bash
git clone https://github.com/alicamdal/vscode-hpc-tutorial.git
cd vscode-hpc-tutorial
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy
sbatch hpc/check_gpu.sbatch
```

On your laptop:

```bash
git clone https://github.com/alicamdal/vscode-hpc-tutorial.git
cd vscode-hpc-tutorial
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows: pip install torch
pip install -r requirements.txt
python demo/vit_demo.py
```

## The model

The demo uses [facebook/deit-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224)
(DeiT-Tiny, 5.7 million parameters, Apache-2.0 license), a Vision Transformer trained on ImageNet-1k.
Any other ViT classifier from Hugging Face can be tried with `--model`.

Reference: H. Touvron et al., "Training data-efficient image transformers & distillation through
attention," *Proc. ICML*, 2021.

## Contact

Ali Camdal, PhD candidate, ITC, University of Twente.
