# LTX-2.5 video model

Downloads Lightricks' open LTX-2.5 video model (video + sound) into ComfyUI's folder layout.

1. Accept the licence at https://huggingface.co/Lightricks/LTX-2.5 and wait for approval.
2. Set your Hugging Face token: `export HF_TOKEN=hf_...` (in Colab, use the Secrets panel). Never commit it.
3. Run:
   ```
   pip install -U huggingface_hub
   python tools/ltx/setup_ltx.py               # small, about 40 GB
   python tools/ltx/setup_ltx.py --size full   # full quality, about 72 GB
   ```
4. To put the files straight into ComfyUI: add `--dest /path/to/ComfyUI/models`.

The `tools/ltx/models/` folder is git-ignored, so the model never gets pushed to GitHub.
Licence: LTX-2 Community License, free while revenue is under $10M a year.
