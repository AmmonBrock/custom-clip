""" Downloads the trained model and uses it to sort images in a folder by their projection between two captions. Make up your own captions or specify a path to your own folder of images. """

from analysis.download_models import download_trained_model_and_processor
from analysis.projection import sort_image_folder
from pathlib import Path
import argparse
import torch

def parse_args():
    """Parse command line arguments.
    --folder: Path to the folder containing images. If not provided, uses data/ammon_pictures.
    --caption1: First caption defining the line.
    --caption2: Second caption defining the line.
    """

    parser = argparse.ArgumentParser(description="Sort images by semantic projection between two captions.")
    parser.add_argument("--folder", type=str, default=None, 
                        help="Path to the folder containing images. If not provided, uses data/ammon_pictures.")
    parser.add_argument("--caption1", type=str, default="A very ugly man", help="First caption defining the line.")
    parser.add_argument("--caption2", type=str, default="An extremely attractive man", help="Second caption defining the line.")
    return parser.parse_args()



if __name__ == "__main__":


    # Check to see if the model has already been downloaded, if not download it
    pretrained_models_dir = Path(__file__).resolve().parent / "analysis" / "pretrained_models"
    model_dir = pretrained_models_dir / "ammonbro" / "clip650r-epoch4"
    if not model_dir.exists():
        print("Downloading trained model...", flush = True)
        download_trained_model_and_processor(model_name = "ammonbro/clip650r-epoch4", save_dir = str(pretrained_models_dir))
        print("Trained model downloaded successfully.", flush = True)

    # command line argument to specify folder of images to sort and captions to define the line
    args = parse_args()
    if args.folder:
        folder_path = Path(args.folder)
    else:
        folder_path = Path(__file__).resolve().parent / "data" / "ammon_pictures"
    

    # Sort images and save a sample display
    print(folder_path, flush = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sort_image_folder(folder_path, args.caption1, args.caption2, model_name="ammonbro/clip650r-epoch4", device=device, max_display_images=20)
    
