"""Download LTX-2.5 (Lightricks' open video model) ready for ComfyUI.

How to run (Colab or any machine with Python):
    pip install -U huggingface_hub
    python tools/ltx/setup_ltx.py                 # small version (~40 GB), default
    python tools/ltx/setup_ltx.py --size full     # full-quality version (~72 GB)
    python tools/ltx/setup_ltx.py --dest /content/ComfyUI/models

Before the first run:
  1. Accept the licence at https://huggingface.co/Lightricks/LTX-2.5
  2. Set your Hugging Face token: export HF_TOKEN=hf_...   (never commit it)

The files land in ComfyUI's folder layout (diffusion_models, text_encoders, vae ...).
The models folder is git-ignored so nothing large is ever pushed to GitHub.
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "Lightricks/LTX-2.5"

SHARED = [
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
]

SIZES = {
    # Smaller 8-bit files: fits Colab disk, still runs well on an H100.
    "small": [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    ],
    # Full-precision files: best quality, needs ~72 GB of disk.
    "full": [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ],
}


def main():
    p = argparse.ArgumentParser(description="Download LTX-2.5 for ComfyUI")
    p.add_argument("--size", choices=SIZES, default="small")
    p.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "models"))
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN first (and accept the licence on the LTX-2.5 Hugging Face page).")

    files = SIZES[args.size] + SHARED
    for f in files:
        print(f"Downloading {f} ...")
        hf_hub_download(REPO, f, local_dir=args.dest)
    print(f"Done. LTX-2.5 ({args.size}) is in {args.dest}")


if __name__ == "__main__":
    main()
