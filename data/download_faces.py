from datasets import load_dataset
from pathlib import Path

def download_utkface(save_dir="./utkface_dataset"):
    """
    Download the UTKFace dataset and save images locally.
    
    Args:
        save_dir: Directory where the dataset will be saved
    """
    print("Loading UTKFace dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("nu-delta/utkface")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {save_path.absolute()}")
    
    # Process each split in the dataset
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} images)...")
        split_dir = save_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Save each image with metadata in filename
        for idx, example in enumerate(split_data):
            # Extract metadata
            age = example.get('age', 'unknown')
            gender = example.get('gender', 'unknown')
            ethnicity = example.get('ethnicity', 'unknown')
            filename = f"{idx:05d}_age{age}_gender{gender}_ethnicity{ethnicity}.jpg"
            filepath = split_dir / filename
            image = example['image']
            image.save(filepath)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Saved {idx + 1}/{len(split_data)} images...")
        
        print(f"  Completed {split_name} split!")
    
    print(f"\n✓ Dataset downloaded successfully to {save_path.absolute()}")
    print(f"  Total splits: {len(dataset)}")
    for split_name in dataset.keys():
        print(f"  - {split_name}: {len(dataset[split_name])} images")