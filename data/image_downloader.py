"""This script is used to download images from URLs specified in TSV files, with support for batch downloading and resuming."""

import requests
import pandas as pd
from PIL import Image
import io
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Image downloader with batch support and resume capability

BATCH_SIZE = 100000


def download_image(index, row, prefix=""):
    try:
        caption = row["caption"]
        url = row["url"]

        response = requests.get(url, timeout=10)
        if response.status_code != 200 or len(response.content) == 0:
            return None, "HTTP error or empty content"

        img = Image.open(io.BytesIO(response.content))
        img.verify()

        filename = f"images/{prefix}{index}.jpg"
        with open(filename, "wb") as f:
            f.write(response.content)

        return (f"{prefix}{index}.jpg", caption), None

    except Exception as e:
        return None, str(e)


def download_with_results_collection(df, start_index=0, batch_size=BATCH_SIZE, prefix=""):
    results = []
    fail_counter = 0
    success_counter = 0

    end_index = min(start_index + batch_size, len(df))
    
    print(f"Downloading images from index {start_index} to {end_index - 1}")
    print(f"Total images to download: {end_index - start_index}")

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Select only the batch we want to download
        df_batch = df.iloc[start_index:end_index]
        
        future_to_index = {
            executor.submit(download_image, index, row, prefix=prefix): index
            for index, row in df_batch.iterrows()
        }

        for future in as_completed(future_to_index):
            result, error = future.result()
            if result:
                results.append(result)
                success_counter += 1
            else:
                fail_counter += 1
            
            # Print progress every 1000 images
            if (success_counter + fail_counter) % 1000 == 0:
                print(f"Progress: {success_counter + fail_counter}/{end_index - start_index} images processed")

    if results:
        with open("dataset.tsv", "a") as f:
            for filename, caption in results:
                f.write(f'{filename}\t{caption}\n')

    print(f"Batch complete - Success: {success_counter}, Fail: {fail_counter}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download images in batches')
    parser.add_argument('--start', type=int, default=0,
                        help=f'Starting index for download (default: 0). Downloads {BATCH_SIZE} images from this point.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Number of images to download in this batch (default: {BATCH_SIZE})')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'both'], default='train',
                        help='Which dataset to download: train, val, or both (default: train)')
    
    args = parser.parse_args()

    os.makedirs("images", exist_ok=True)
    train_path = "data/Train-GCC-training.tsv"
    val_path = "data/Validation_GCC-1.1.0-Validation.tsv"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Training or validation TSV file not found in the data directory. Make sure the following paths exist:\n"
            f"{train_path}\n"
            f"{val_path}\n"
        )

    if args.dataset in ['train', 'both']:
        print(f"\n{'='*60}")
        print("Processing TRAINING dataset")
        print(f"{'='*60}")
        df = pd.read_csv(train_path, sep="\t",
                         header=None, names=["caption", "url"])
        print(f"Total training images available: {len(df)}")
        download_with_results_collection(df, start_index=args.start, batch_size=args.batch_size)

    if args.dataset in ['val', 'both']:
        print(f"\n{'='*60}")
        print("Processing VALIDATION dataset")
        print(f"{'='*60}")
        df_validation = pd.read_csv(val_path, sep="\t",
                                    header=None, names=["caption", "url"])
        print(f"Total validation images available: {len(df_validation)}")
        download_with_results_collection(df_validation, start_index=args.start, 
                                        batch_size=args.batch_size, prefix="val_")
