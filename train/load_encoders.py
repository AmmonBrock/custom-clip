import torch
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)


import os
from PIL import Image
from train.data_handling_webdataset import create_dataloader



def get_model_and_processor(device = None, text_encoder = "bert", vision_encoder = "vit"):
    """Load a dual encoder model with specified text and vision encoders along with their tokenizers and processors.
     Args:
        device (str, optional): Device to load the model onto. Defaults to None, which
                                automatically selects "cuda" if available, otherwise "cpu".
        text_encoder (str, optional): Type of text encoder to use. Defaults to "bert". Options: "bert", "neobert", "roberta"
        vision_encoder (str, optional): Type of vision encoder to use. Defaults to "vit". Options: "resnet", "vit"
    Returns:
        VisionTextDualEncoderModel: The loaded dual encoder model.
        processor: The combined processor for text and images.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder_dict = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base"
    }
    vision_encoder_dict = {
        "vit": "google/vit-base-patch16-224",
        "beit": "microsoft/beit-base-patch16-224-pt22k",
        # "resnet": "microsoft/resnet-50" # Not compatible with dual encoder model
    }

    assert text_encoder in text_encoder_dict, f"Unsupported text encoder: {text_encoder}"
    assert vision_encoder in vision_encoder_dict, f"Unsupported vision encoder: {vision_encoder}"

    text_model_path = os.path.join("./pretrained_models", text_encoder)
    vision_model_path = os.path.join("./pretrained_models", vision_encoder)


    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only = True)
    image_processor = AutoImageProcessor.from_pretrained(vision_model_path, local_files_only = True, use_fast=True)
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    model = VisionTextDualEncoderModel.from_vision_text_pretrained(vision_model_path, text_model_path, local_files_only = True)
    model.to(device)


    return model, processor