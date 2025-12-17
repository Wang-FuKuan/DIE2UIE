# DIE2UIE

Underwater Image Enhancement via Degradation Information Extraction and Guidance

This repository provides training code and setup instructions for DIE2UIE. The method extracts degradation information and uses it as guidance for underwater image enhancement.


## Contents
- Environment
- Dataset
- Preparation
- Training
- Acknowledgements
- Citation

## Environment

Dependencies are listed in requirements.txt.

Recommended setup (conda):

```bash
git clone <your-repo-url>
cd DIE2UIE

conda create -n die2uie python=3.10 -y
conda activate die2uie

pip install -r requirements.txt
```

## Dataset (UIEBD)

Download UIEBD dataset:
- https://drive.google.com/drive/folders/1xcU3_SOgAW9gDmhj9XNE4HKAikLGnwq2?usp=sharing

Place the dataset anywhere on your disk and configure its root path in train_DIE.py (see Training section).

Suggested dataset layout (adjust to your actual UIEBD folder structure):

```text
UIEBD/
  train/
  val/
```

## Preparation

### Step 1: Degradation prompt initialization (CLIP-LIT style)

This project initializes degradation prompts using the dataset splits provided in:

- dataset/Degradation Prompt Learning/

and follows the prompt learning strategy in CLIP-LIT:
- https://github.com/ZhexinLiang/CLIP-LIT

What you need to do
1. Use the split files in dataset/Degradation Prompt Learning/ to construct the prompt-learning dataset.
2. Run prompt learning following CLIP-LIT.
3. Export the learned degradation prompt weights to a file (example path: weights/degradation_prompt_init.pth).
4. Configure train_DIE.py to load this prompt initialization checkpoint.


### Step 2: Pretrained RAM weights

This project uses Recognize Anything Model (RAM) as a pretrained component.

Reference repository:
- https://github.com/xinyu1205/recognize-anything

Default checkpoint used by this repo:
- ram_swin_large_14m.pth

Suggested placement:

```text
DIE2UIE/
  weights/
    ram_swin_large_14m.pth
    degradation_prompt_init.pth
```

Then set the corresponding paths in train_DIE.py.

## Training

Training entry:
- train_DIE.py

Minimal run:

```bash
python train_DIE.py
```

Before running, open train_DIE.py and configure (at minimum):
- UIEBD dataset root path
- RAM checkpoint path (ram_swin_large_14m.pth)
- Degradation prompt init checkpoint path (from Step 1)
- Output directory for checkpoints/logs
- Training hyperparameters you want to change (batch size, learning rate, epochs, GPU id, etc.)

Acknowledgements

This work references or is inspired by the following projects:
- CLIP-LIT: https://github.com/ZhexinLiang/CLIP-LIT
- HCLR-Net: https://github.com/zhoujingchun03/HCLR-Net
- Recognize Anything (RAM): https://github.com/xinyu1205/recognize-anything
- PromptIR: https://github.com/va1shn9v/PromptIR

Citation

If you find this repository useful, please cite the corresponding paper:

```bibtex
@article{die2uie,
  title={Underwater Image Enhancement via Degradation Information Extraction and Guidance},
  author={},
  journal={},
  year={}
}
```
