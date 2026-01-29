# CLIP-Style Vision-Language Model

A custom implementation of contrastive vision-language pretraining, aligning pretrained text and image encoders through CLIP-style contrastive loss on 1.2 million image-caption pairs. We also include a fun application that uses 2 captions to sort a folder of images along the line defined by the captions.

## Quick Start (Demo)

The easiest way to use our trained model is through the demo script:

```bash
# Create virtual environment and install dependencies
uv sync

# Run the demo
uv run demo.py

# Or for custom image folder or semantic lines defined by 2 captions
uv run demo.py --folder "path/to/image_folder" --caption1 "something dumb" --caption2 "something cool"
```

The demo script will automatically download our trained model and show you how to use it for a task like semantic image sorting.

## Project Structure

```
├── demo.py              # Quick start demo (recommended entry point)
├── train/               # Training code and configuration
├── analysis/            # Model evaluation and CLIP comparison
└── data/                # Data downloading, cleaning, and preprocessing
```

### Code Organization

- **`demo.py`**: Fully self-contained demo showing how to download and use our trained model. Start here!
- **`train/`**: Code used for training the model on 1.2M image-caption pairs using distributed compute
- **`analysis/`**: Evaluation scripts and comparisons against the original CLIP model
- **`data/`**: Pipeline for downloading, cleaning, and organizing training data into WebDataset format

## Important Note

The `train/`, `analysis/`, and `data/` directories are provided **for reference and transparency** to show our complete methodology. They are not optimized for easy replication, as the full training pipeline required:

- 1.2 million image-caption pairs downloaded in batches
- University supercomputer resources for distributed training
- Conversion to WebDataset format for efficient data loading

**If you want to use our model**, stick with `demo.py` - it shows how to download the model from huggingface and extract

**If you want to understand our methodology**, the training and analysis code provides full implementation details.

## Model Details

- **Architecture**: Pretrained text encoder (BERT) + pretrained image encoder (ViT) aligned via contrastive learning
- **Training Data**: 1.2 million image-caption pairs
- **Training Objective**: CLIP-style contrastive loss
- **Key Features**: Zero-shot classification, semantic image search, cross-modal retrieval

## Applications

Our trained model supports various tasks including:
- Semantic image sorting along conceptual dimensions
- Zero-shot image classification
- Text-to-image retrieval
- Image-to-text retrieval

## Contributors
- Ammon Brock
- Matthew Gabbitas
- Yu-Hsien Jen
