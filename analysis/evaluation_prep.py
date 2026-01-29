"""The evaluation.py file depends on previously downloaded datasets and models."""

from datasets import load_dataset
from transformers import AutoModel, CLIPProcessor   
from pathlib import Path
from data.download_faces import download_utkface
script_path = Path(__file__).resolve()



# Download CIFAR-10 small dataset to measure zero-shot classification performance
cifar_dir = script_path.parent.parent / "data" / "cifar10-small"
if not cifar_dir.exists():
    dataset = load_dataset("Technical1113/CIFAR10-small", split="test")
    dataset.save_to_disk(cifar_dir)

# Download utkface dataset
utkface_dir = script_path.parent.parent / "data" / "utkface_dataset"
if not utkface_dir.exists():
    download_utkface(save_dir = str(utkface_dir))

# Download openai - clip for comparison
clip_dir = script_path.parent / "pretrained_models" / "clip-vit-large-patch14"
if not clip_dir.exists():
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

    model.save_pretrained(clip_dir)
    processor.save_pretrained(clip_dir)


