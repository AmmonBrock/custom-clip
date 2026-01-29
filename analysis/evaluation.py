"""
1. Zero-shot classification accuracy: Evaluate the model's ability to classify images
2. Recall@k: Measure how often the correct image-caption pair is among the top-k retrieved results in a retrieval task.
3. Median rank: Calculate the median rank of the correct caption when retrieving captions for a given image, or vice versa.
4. Zero-shot age ranking correlation: Novel metric to assess how well the model can sort images on a semantically defined axis (e.g., age) without explicit training for that task.

This script utilizes several outside datasets that need to be downloaded prior to running evaluation: 
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, CLIPProcessor
import faiss
import numpy as np
from faiss_setup import get_image_embedding
import os
import pandas as pd
from analysis.load_models import get_trained_model_and_processor, get_pretrained_clip
from analysis.projection import ProjectImagesText, load_utkface_images
from scipy.stats import spearmanr




def zero_shot_classify(model, processor, images, label_options):
    """Perform zero-shot classification on a batch of images

    Args:
        model: The trained VisionTextDualEncoderModel
        processor: The VisionTextDualEncoderProcessor
        images: list of RGB PIL images
        label_options: list of strings, the class labels to classify into
    Returns:
       tensor of shape (num_images,) with the predicted class indices
    """


    device = model.device
    print(device, flush = True)

    # Make the class label feel more like an image caption
    label_captions = [f"a photo of a {label}" for label in label_options]

    # Get text embeddings for each label option
    text_inputs = processor.tokenizer(
        label_captions,
        return_tensors="pt",
        padding=True,
        truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    print("Moved text inputs to device", flush = True)
    # Get text embeddings directly from the text encoder
    with torch.no_grad():  # Use no_grad if you're doing inference
        label_embeddings = model.get_text_features(**text_inputs)


    print("Got label embeddings", flush = True)
    # Get image embeddings for each image
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    print("Processed image inputs", flush = True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)


    # Normalize embeddings so we can compute similarity via dot product

    image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=1)
    label_embeddings_norm = F.normalize(label_embeddings, p=2, dim=1)
    print("Normalized embeddings", flush = True)
    classes = torch.matmul(image_embeddings_norm, label_embeddings_norm.T).argmax(dim=1)
    return classes

def evaluate_with_zero_shot(model, processor):

    # Load dataset from local disk
    try:
        dataset = load_from_disk("data/cifar10-small")
    except FileNotFoundError:
        print("Dataset not found on disk. Please run evaluation_prep.py on a login node to download the dataset")
        return

    label_options = dataset.features["label"].names
    labels = dataset['label']

    images = [img.convert("RGB") for img in dataset['image']]

    predicted_classes = zero_shot_classify(model, processor, images, label_options)
    accuracy = (predicted_classes.cpu() == torch.tensor(labels)).float().mean().item()
    print(f"Zero-shot classification accuracy on CIFAR-10 test set: {accuracy*100:.2f}%")

    return accuracy
    
def get_clip_zero_shot(device = "cpu"):
    try:
        model = AutoModel.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True)
        processor = CLIPProcessor.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True, use_fast=True)
    except: 
        print("Must run evaluation_prep.py on a login node to download the CLIP model and processor")
        return
    model.to(device)
    return evaluate_with_zero_shot(model, processor)



def recall_at_k(model, processor, k, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index = False):
    """Compute Recall@k for image-caption retrieval on holdout data
    Args:
        model: The trained VisionTextDualEncoderModel
        processor: The VisionTextDualEncoderProcessor
        k: int, the 'k' in Recall@k
        faiss_index_file: str, path to the FAISS index file containing image embeddings
        image_names_file: str, path to the numpy file containing image names corresponding to the FAISS index
        holdout_df: pd.DataFrame with columns ["caption", "path"] for holdout data
        rewrite_index: bool, whether to recreate the FAISS index from the image folder
    Returns:
        recall_at_k: float, the Recall@k value  
    """

    device = model.device
    model.eval()

 

    if (not os.path.exists(faiss_index_file)) or rewrite_index:

        print("Creating FAISS index from image folder...", flush = True)

        embeddings, image_names = get_image_embedding(image_folder, model, processor, device)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.reshape(-1, d))
        faiss.write_index(index, faiss_index_file)
        image_names = np.array(image_names)
        np.save(image_names_file, image_names)
        print("FAISS index and image names saved.", flush = True)
    else:
        index = faiss.read_index(faiss_index_file)
        image_names = np.load(image_names_file)

    captions = holdout_df['caption'].tolist()
    names = holdout_df['path'].tolist()

    # Every value in names should match an image name in image_names
    image_names_list = image_names.tolist()
    names_indices = np.array([image_names_list.index(image) for image in names])

    print("Got names_indices", flush = True)




    query_embedding = processor(text = captions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = model.get_text_features(**query_embedding).cpu().numpy()

    print("Searching for nerest neighbors...", flush = True)
    D, I = index.search(query_embedding, k)

    recall_at_k = np.mean(np.any(I == names_indices.reshape(-1,1), axis=1)) # Gets recall@k

    return recall_at_k

def median_rank(model, processor, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index = False):
    device = model.device
    model.eval()

 

    if (not os.path.exists(faiss_index_file)) or rewrite_index:

        print("Creating FAISS index from image folder...", flush = True)

        embeddings, image_names = get_image_embedding(image_folder, model, processor, device)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.reshape(-1, d))
        faiss.write_index(index, "image_index.faiss")
        image_names = np.array(image_names)
        np.save("image_names.npy", image_names)
        print("FAISS index and image names saved.", flush = True)
    else:
        index = faiss.read_index(faiss_index_file)
        image_names = np.load(image_names_file)

    captions = holdout_df['caption'].tolist()
    names = holdout_df['path'].tolist()

    # Every value in names should match an image name in image_names
    image_names_list = image_names.tolist()
    names_indices = np.array([image_names_list.index(image) for image in names])

    print("Got names_indices", flush = True)

    query_embedding = processor(text = captions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = model.get_text_features(**query_embedding).cpu().numpy()
    
    D, I = index.search(query_embedding, len(image_names))
    ranks = [np.where(I[i] == names_indices[i])[0][0] + 1 for i in range(len(names_indices))]
    median_rank_value = np.median(ranks)
    return median_rank_value

def get_clip_recall_at_k(device, k, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index = False):

    try:
        model = AutoModel.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True)
        processor = CLIPProcessor.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True, use_fast=True)
    except: 
        print("Must run evaluation_prep.py on a login node to download the CLIP model and processor")
        return
    model.to(device)

    return recall_at_k(model, processor, k, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index)

def get_clip_median_rank(device, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index = False):
    
    try:
        model = AutoModel.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True)
        processor = CLIPProcessor.from_pretrained("pretrained_models/clip-vit-large-patch14", local_files_only = True, use_fast=True)
    except: 
        print("Must run evaluation_prep.py on a login node to download the CLIP model and processor")
        return
    model.to(device)

    return median_rank(model, processor, image_folder, holdout_df, faiss_index_file, image_names_file, rewrite_index)

def get_zero_shot_age_ranking(model_name, num_images = 1000):
    # Make checkpoint_path = None if you want pretrained clip
    np.random.seed(42)
    images, ages, genders, ethnicities = load_utkface_images("data/utkface", split="train", max_images=num_images)
    pit = ProjectImagesText(model_name=model_name, device="cuda")
    caption1 = "A very young baby"
    caption2 = "An extremely old person"
    sorted_images, sorted_indices = pit.sort_from_caption1_to_caption2(caption1, caption2, images)
    sorted_ages = np.array(ages)[sorted_indices]

    correlation, p_value = spearmanr(sorted_ages, range(len(sorted_ages)))

    del pit
    del images
    del sorted_images
    torch.cuda.empty_cache()

    return correlation, p_value


    
if __name__ == "__main__":

    device = "cuda"
    model, processor = get_trained_model_and_processor(device = device, model_name = "ammonbro/clip650r-epoch4")
    model.eval()
    print("Loaded trained model", flush = True)

    
    # Load holdout dataframe
    try:
        df = pd.read_csv("captions.tsv", sep = "\t").sample(n=2000, random_state=42).reset_index(drop=True)
    except:
        print("captions.tsv not found. The captions.tsv file should contain two columns: 'path' and 'caption' (e.g., '0.jpg'    'a row of old fashioned cars.'). Please see https://ai.google.com/research/ConceptualCaptions/ for more information.", flush = True)
        exit(1)
    if Path('data/validation').exists() == False:
        print("data/validation folder not found. Please create a data/validation folder containing the images referenced in captions.tsv", flush = True)
        exit(1)



    # Recall @ 10
    recall_at_10 = recall_at_k(model, processor, k = 10, image_folder = "data/validation", holdout_df = df, faiss_index_file="epoch_4_image_index.faiss", image_names_file = "epoch_4_image_names.npy", rewrite_index = False)
    print("Our model Recall@10:", recall_at_10, flush = True)
    clip_recall_at_10 = get_clip_recall_at_k(device = "cuda", k = 10, image_folder = "data/validation", holdout_df = df, faiss_index_file = "clip_image_index.faiss", image_names_file = "clip_image_names.npy", rewrite_index = False)
    print("CLIP Recall@10:", clip_recall_at_10, flush = True)


    # Median Rank
    custom_median_rank = median_rank(model, processor, image_folder = "data/validation", holdout_df = df, faiss_index_file="epoch_4_image_index.faiss", image_names_file = "epoch_4_image_names.npy", rewrite_index = False)
    custom_median_rank = int(custom_median_rank)
    print("Our model Median Rank:", custom_median_rank, flush = True)
    clip_median_rank = get_clip_median_rank(device = "cuda", image_folder = "data/validation", holdout_df = df, faiss_index_file = "clip_image_index.faiss", image_names_file = "clip_image_names.npy", rewrite_index = False)
    clip_median_rank = int(clip_median_rank)
    print("CLIP Median Rank:", clip_median_rank, flush = True)


    # Zero-shot classification accuracy
    zero_shot_classify_accuracy = evaluate_with_zero_shot(model, processor)
    print("Our model Zero-shot classification accuracy on CIFAR-10:", zero_shot_classify_accuracy, flush = True)
    zero_shot_clip = get_clip_zero_shot(device = "cuda")
    print("CLIP Zero-shot classification accuracy on CIFAR-10:", zero_shot_clip, flush = True)


    # Age ranking correlation
    custom_age_rank_corr, custom_age_rank_p = get_zero_shot_age_ranking("ammonbro/clip650r-epoch4", num_images=1000)
    print(f"Our model Zero-shot age ranking Spearman correlation: {custom_age_rank_corr}, p-value: {custom_age_rank_p}", flush=True)
    torch.cuda.empty_cache()
    clip_age_rank_corr, clip_age_rank_p = get_zero_shot_age_ranking(None, num_images=1000)
    print(f"CLIP Zero-shot age ranking Spearman correlation: {clip_age_rank_corr}, p-value: {clip_age_rank_p}", flush=True)




    print("\n\n\n Summary of Evaluation Metrics:")

    our_index = faiss.read_index("epoch_4_image_index.faiss")
    clip_index = faiss.read_index("clip_image_index.faiss")
    print(f"Our model index size: {our_index.ntotal} images")
    print(f"CLIP index size: {clip_index.ntotal} images")

    model_comparison_df = pd.DataFrame({
        "Model": ["Our Model", "CLIP"],
        "Recall@10": [recall_at_10, clip_recall_at_10],
        "Median Rank": [custom_median_rank, clip_median_rank],
        "Index Size": [our_index.ntotal, clip_index.ntotal],
        "Query Sample Size": [len(df), len(df)],
        "Zero-shot Classification Accuracy": [zero_shot_classify_accuracy, zero_shot_clip],
        "Age Ranking Spearman Correlation": [custom_age_rank_corr, clip_age_rank_corr],
        "Age Ranking Spearman p-value": [custom_age_rank_p, clip_age_rank_p]
    })
    print(model_comparison_df)



    model_comparison_df.to_csv("model_evaluation_summary.csv", index=False)


