from transformers import AutoTokenizer, AutoImageProcessor, AutoModel, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import os
import torch


def download_and_save_models(save_dir="./pretrained_encoders"):
    """Download only the base models needed."""
    
    text_encoder_dict = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base"
    }
    vision_encoder_dict = {
        "vit": "google/vit-base-patch16-224",
        "beit": "microsoft/beit-base-patch16-224-pt22k",
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Download text encoders and tokenizers
    for name, model_id in text_encoder_dict.items():
        print(f"Downloading {name}...")
        model_path = os.path.join(save_dir, name)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)

        model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(model_path)
        print(f"Saved to {model_path}")
    
    # Download vision encoders and image processors
    for name, model_id in vision_encoder_dict.items():
        print(f"Downloading {name}...")
        model_path = os.path.join(save_dir, name)
        
        image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        image_processor.save_pretrained(model_path)
        
        model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(model_path)
        print(f"Saved to {model_path}")

if __name__ == "__main__":
    download_and_save_models()
    print("Base encoders downloaded successfully!")


