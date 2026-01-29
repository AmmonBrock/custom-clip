from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor, AutoModel, CLIPProcessor 
import os
from pathlib import Path

def download_trained_model_and_processor(model_name = "ammonbro/clip650r-epoch4", save_dir = "./pretrained_models"):
    """Load a dual encoder model from the Hugging Face Hub. Save to local directory.
     Args:
        model_name (str, optional): Name of the model on the Hugging Face Hub. Defaults to "ammonbro/clip650r-epoch4".
        save_dir (str, optional): Directory to save the model and processor. Defaults to "./pretrained_models".
    Returns:
        VisionTextDualEncoderModel: The loaded dual encoder model.
        processor: The combined processor for text and images.
    """

    model = VisionTextDualEncoderModel.from_pretrained(model_name)
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_name)

    model_path = Path(save_dir) / model_name
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    print(f"Saved model and processor to {model_path}")


def download_pretrained_clip():
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    model.save_pretrained("pretrained_models/clip-vit-large-patch14")
    processor.save_pretrained("pretrained_models/clip-vit-large-patch14")

if __name__ == "__main__":
    download_trained_model_and_processor()
    print("Trained model downloaded successfully!", flush = True)
    download_pretrained_clip()
    print("Pretrained CLIP model downloaded successfully!", flush = True)
