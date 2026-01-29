# File for pytorch models that will be used in training
import torch
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoModel,
    CLIPProcessor
)
import os
from pathlib import Path

def get_trained_model_and_processor(device = None, model_name = "ammonbro/clip650r-epoch4"):
    """Load a trained dual encoder model from local ./pretrained_models directory along with its processor.
     Args:
        device (str, optional): Device to load the model onto. Defaults to None, which
                                automatically selects "cuda" if available, otherwise "cpu".
        model_name (str, optional): Name of the model directory in ./pretrained_models. Defaults to "ammonbro/clip650r-epoch4".
     Returns:
        VisionTextDualEncoderModel: The loaded dual encoder model.
        VisionTextDualEncoderProcessor: The combined processor for text and images.
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(__file__).resolve().parent / "pretrained_models" / model_name

    # Check if model path exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist. Please download the model first. (download_models.py)")

    

    model = VisionTextDualEncoderModel.from_pretrained(model_path, local_files_only = True)
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_path, local_files_only = True)

    model.to(device)

    return model, processor

def get_pretrained_clip(device = None):
    """Load the pretrained CLIP model from local ./pretrained_models directory along with its processor.
     Args:
        device (str, optional): Device to load the model onto. Defaults to None, which
                                automatically selects "cuda" if available, otherwise "cpu".
     Returns:
        AutoModel: The loaded CLIP model.
        CLIPProcessor: The processor for CLIP.
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(__file__).resolve().parent / "pretrained_models" / "clip-vit-large-patch14"

    # Check if model path exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist. Please download the model first. (download_models.py)")

    model = AutoModel.from_pretrained(model_path, local_files_only = True)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only = True)

    model.to(device)

    return model, processor


            

    
