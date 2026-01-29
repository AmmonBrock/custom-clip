"""Use this script to sort images based on a semantic line defined by two text prompts."""

import torch
from datasets import load_dataset
from pathlib import Path
import os
import torchvision.transforms as T
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPProcessor
from scipy.stats import spearmanr
from analysis.load_models import get_trained_model_and_processor, get_pretrained_clip
from matplotlib import pyplot as plt


class ProjectImagesText:
    def __init__(self, model_name, device, batch_size=32):
        """Initialize the ProjectImagesText class with a dual encoder model and processor.
            Args:
                model_name (str): Name of the model to load. Use "clip-vit-large-patch14" for pretrained CLIP.
                device (str): Device to load the model onto ("cuda" or "cpu").
                batch_size (int): Batch size for processing images.
        """
        if model_name == "clip-vit-large-patch14":
            self.model, self.processor = get_pretrained_clip(device=device)
        else:
            self.model, self.processor = get_trained_model_and_processor(device=device, model_name=model_name)
        
        self.model.to(device)
        self.device = device
        self.model.eval()
        self.batch_size = batch_size
    
    def _get_line_for_projection(self, caption1, caption2):
        """Compute the normalized line vector between two caption embeddings.
            Args:
                caption1 (str): First caption.
                caption2 (str): Second caption.
            Returns:
                np.ndarray: Normalized line vector between the two caption embeddings.
        """

        text_inputs = self.processor.tokenizer(
            [caption1, caption2],
            return_tensors="pt",
            padding=True,
            truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            label_embeddings = self.model.get_text_features(**text_inputs)[1]
        
        # Compute the difference line and normalize
        diff_line = label_embeddings[0] - label_embeddings[1]
        diff_line = diff_line / diff_line.norm(dim=0, keepdim=True)
        return diff_line.cpu().numpy()
    
    def project_images(self, caption1, caption2, images):
        """Project images onto the line between two captions, processing in batches.
            Args:
                caption1 (str): First caption defining the line.
                caption2 (str): Second caption defining the line.
                images (list of PIL.Image or tensors): List of images to project.
            Returns:
                np.ndarray: Projections of images onto the line.
        """

        line_vector = self._get_line_for_projection(caption1, caption2)
        
        all_projections = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:min(i+self.batch_size, len(images))]
            
            image_inputs = self.processor.image_processor(images=batch_images, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**image_inputs)[1]
            
            # Normalize image embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
            
            # Dot product with line_vector
            projections = torch.matmul(image_embeddings.cpu(), torch.tensor(line_vector).unsqueeze(1)).squeeze(1)
            all_projections.append(projections.numpy())
            
            # Clear GPU cache after each batch
            del image_inputs, image_embeddings
            torch.cuda.empty_cache()
        
        return np.concatenate(all_projections)
    
    def sort_from_caption1_to_caption2(self, caption1, caption2, images):
        """Sort images based on their projection from caption1 to caption2.
            Args:
                caption1 (str): First caption defining the line.
                caption2 (str): Second caption defining the line.
                images (list of PIL.Image or tensors): List of images to sort. 
            Returns:
                sorted_images (list): Images sorted based on projection.
                sorted_indices (np.ndarray): Indices that would sort the images.
        """
         
        projections = self.project_images(caption1, caption2, images)
        sorted_indices = projections.argsort()[::-1]
        return np.array(images)[sorted_indices], sorted_indices



def load_utkface_images(dataset_dir="./utkface_dataset", split="train", max_images=None, apply_transform=True):
    """
    Load images from saved UTKFace dataset directory.
    
    Args:
        dataset_dir: Directory where dataset was saved
        split: Which split to load (e.g., 'train', 'test')
        max_images: Optional limit on number of images to load
        apply_transform: If True, apply preprocessing transforms to images
        
    Returns:
        images: List of PIL Images (or transformed tensors if apply_transform=True)
        age: List of ages
        gender: List of genders (0 for Male, 1 for Female)
        ethnicity: List of ethnicities
    """
    # Define the preprocessing transform
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    split_dir = Path(dataset_dir) / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    # Get all jpg files
    image_files = sorted(split_dir.glob("*.jpg"))

    
    if max_images:
        if max_images < len(image_files):
            np.random.seed(42)  # For reproducibility
            random_indices = np.random.choice(len(image_files), max_images, replace=False)
            image_files = [image_files[i] for i in random_indices]
    
    print(f"Loading {len(image_files)} images from {split} split...")
    
    images = []
    age = []
    gender = [] # 0 for Male and 1 for Female
    ethnicity = [] 
    
    for img_path in image_files:
        img = Image.open(img_path)
        
        if apply_transform:
            img = transform(img)
        
        images.append(img)
        
        # Parse metadata from filename
        # Format: 00000_age25_gender1_ethnicity2.jpg
        parts = img_path.stem.split('_')
        age.append(int(parts[1].replace('age', '')) if len(parts) > 1 else None)
        g = parts[2].replace('gender', '') if len(parts) > 2 else None
        gender.append(0 if g == "Male" else 1)
        ethnicity.append(parts[3].replace('ethnicity', '') if len(parts) > 3 else None)
    
    print(f"Loaded {len(images)} images")
    return images, age, gender, ethnicity

def load_images_from_folder(folder_path:str = "data/ammon_pictures", apply_transform=True):
    """ Load images from a specified folder.
    Converts other image types to jpg if possible.
    folder_path: Path to the folder containing images
    apply_transform: Whether to apply preprocessing transforms
    """

    # Define the preprocessing transform
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    # Supported image formats
    supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.tiff')
    image_files = []
    folder = Path(folder_path)
    
    # Collect all supported image files
    for pattern in supported_formats:
        image_files.extend(folder.glob(pattern))
        image_files.extend(folder.glob(pattern.upper()))  # Handle uppercase extensions
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # For images with alpha channel, convert to RGB
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                # Convert other modes (like grayscale) to RGB
                img = img.convert('RGB')
            
            if apply_transform:
                img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue
    
    return images


def evaluate_age_ranking(sorted_indices, ages, images):
    sorted_ages = np.array(ages)[sorted_indices]
    correlation, p_value = spearmanr(sorted_ages, range(len(sorted_ages)))
    percentile_averages = [np.mean(sorted_ages[int(len(sorted_ages)*i/10):int(len(sorted_ages)*(i+1)/10)]) for i in range(10)]
    # Bar chart percentile averages 
    import matplotlib.pyplot as plt
    plt.bar(((1 + np.arange(10))).astype(str), percentile_averages)
    plt.xlabel("Decile")
    plt.ylabel("Mean Age")
    plt.title("Mean Age by Decile in Projection Ranking")
    plt.savefig("age_percentiles2.png")
    print(f"Spearman correlation between projected ranking and age: {correlation:.4f} (p-value: {p_value:.4e})", flush = True)
    return correlation, p_value

def show_sorted_images(display_images, display_path):
    # 1. Use gridspec_kw to force zero spacing at the start
    fig, axs = plt.subplots(
        1, len(display_images), 
        figsize=(5*len(display_images), 5),
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )

    # Handle the case where there's only one image (axs won't be a list)
    if len(display_images) == 1:
        axs = [axs]

    for i, img in enumerate(display_images):
        axs[i].imshow(img)
        axs[i].axis('off')

    # 2. Use bbox_inches='tight' and pad_inches=0 during save
    # This is the "secret sauce" that removes the outer white border
    plt.savefig(display_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def sort_image_folder(folder_name, caption1, caption2, model_name, device = "cuda", max_display_images = 20):
    images = load_images_from_folder(Path(folder_name), apply_transform=True)
    pit = ProjectImagesText(model_name=model_name, device=device)
    sorted_images, sorted_indices = pit.sort_from_caption1_to_caption2(caption1, caption2, images)
    display_images = []
    if len(sorted_images) > max_display_images:
        # Sample from percentiles for display
        for i in range(max_display_images):
            start_idx = int(len(sorted_images)*i/max_display_images)
            end_idx = int(len(sorted_images)*(i+1)/max_display_images)
            random_idx = np.random.randint(start_idx, end_idx)
            arr = sorted_images[random_idx]
            img_viewable = (arr - arr.min()) / (arr.max() - arr.min())
            img_viewable = img_viewable.transpose(1,2,0)
            img = Image.fromarray((img_viewable*255).astype(np.uint8))
            display_images.append(img)
    else:
        # display all images
        for i in range(len(sorted_images)):
            arr = sorted_images[i]
            img_viewable = (arr - arr.min()) / (arr.max() - arr.min())
            img_viewable = img_viewable.transpose(1,2,0)
            img = Image.fromarray((img_viewable*255).astype(np.uint8))
            display_images.append(img)
    
    show_sorted_images(display_images, f"{caption1}_to_{caption2}.png")
    return sorted_images, sorted_indices

