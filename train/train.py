import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys
import argparse

from train.load_encoders import get_model_and_processor
from train.config_schema import CLIPConfig, read_config
from train.data_handling_webdataset import create_webdataset_dataloader

def get_dataloaders(data_config):
    train_dataloader = create_webdataset_dataloader(
        shard_pattern=data_config.train_shard_pattern,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers
    )
    val_dataloader = create_webdataset_dataloader(
        shard_pattern=data_config.val_shard_pattern,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers
    )
    
    # Calculate lengths
    train_length = calculate_dataloader_length(
        data_config.train_shard_pattern,
        data_config.batch_size,
        samples_per_shard=20000
    )
    val_length = calculate_dataloader_length(
        data_config.val_shard_pattern,
        data_config.batch_size,
        samples_per_shard=20000
    )
    
    return train_dataloader, val_dataloader, train_length, val_length


def initialize_model(config, checkpoint=None):
    device = config.device

    if checkpoint:
        model, processor = get_model_and_processor(
            device=device,
            text_encoder=config.model.text_model,
            vision_encoder=config.model.vision_model
        )
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        global_step = checkpoint_data.get('global_step', 0)
        start_epoch = checkpoint_data.get('epoch', 0) + 1
        return model, processor, global_step, start_epoch
    else:
        model, processor = get_model_and_processor(
            device=device,
            text_encoder=config.model.text_model,
            vision_encoder=config.model.vision_model
        )
        return model, processor, 0, 0


def save_model_checkpoint(model, optimizer, epoch, global_step, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    torch.save(checkpoint, path)


def logd(*args, **kwargs):
    do_output = True
    if do_output:
        print(*args, **kwargs)
        sys.stdout.flush()


def get_output_dirs(config, resume=False):
    base_output_dir = './runs'
    run_dir = os.path.join(base_output_dir, config.logging.run_dir)

    logs_dir = os.path.join(run_dir, 'logs')
    tensorboard_dir = os.path.join(logs_dir, 'tensorboard')
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)
    if os.path.exists(run_dir) and not resume:
        error_msg = f"Run directory '{run_dir}' already exists. Please set a different 'run_dir' in the config, set the '--resume' flag to continue training from the last checkpoint in this run dir, or manually delete this run directory if you want to overwrite it."
        raise ValueError(error_msg)
    else:
        os.makedirs(run_dir, exist_ok=True)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir, exist_ok=True)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    if config.logging.stdout_log:
        stdout_log_path = os.path.join(logs_dir, 'stdout.txt')
        sys.stdout = open(stdout_log_path, 'a')
        tqdm.disable = True
    if config.logging.stderr_log:
        stderr_log_path = os.path.join(logs_dir, 'stderr.txt')
        sys.stderr = open(stderr_log_path, 'a')

    return {'run': run_dir,
            'logs': logs_dir,
            'tensorboard': tensorboard_dir,
            'checkpoints': checkpoints_dir}

def calculate_dataloader_length(shard_pattern, batch_size, samples_per_shard=20000):
    """Calculate the number of batches in a WebDataset dataloader."""
    import glob
    import re
    
    if isinstance(shard_pattern, list):
        num_shards = len(shard_pattern)
    elif isinstance(shard_pattern, str):
        if '*' in shard_pattern:
            shards = glob.glob(shard_pattern)
            num_shards = len(shards)
        elif '{' in shard_pattern:
            # For brace expansion like {000000..000065}
            match = re.search(r'\{(\d+)\.\.(\d+)\}', shard_pattern)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                num_shards = end - start + 1
            else:
                raise ValueError(f"Could not parse brace expansion: {shard_pattern}")
        else:
            num_shards = 1
    else:
        raise ValueError(f"Unsupported shard_pattern type: {type(shard_pattern)}")
    
    total_samples = num_shards * samples_per_shard
    num_batches = total_samples // batch_size
    
    return num_batches

def train(config, resume=False, checkpoint=None):
    output_dirs = get_output_dirs(config, resume=resume)
    logd("Preparing dataloaders...")
    train_loader, val_loader, train_length, val_length = get_dataloaders(config.data)
    logd(f"✅ Done.")
    logd("Initializing models...")
    if resume:
        if not checkpoint:
            # Find latest checkpoint
            checkpoint_files = [f for f in os.listdir(
                output_dirs['checkpoints']) if f.endswith('.pt')]
            if not checkpoint_files:
                raise ValueError(
                    "No checkpoint files found in checkpoints directory.")
            latest_checkpoint = max(
                checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(
                output_dirs['checkpoints'], latest_checkpoint)
            logd(f"Resuming from latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = checkpoint
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint file '{checkpoint_path}' does not exist.")
            logd(f"Resuming from provided checkpoint: {checkpoint_path}")
        model, processor, global_step, start_epoch = initialize_model(
            config, checkpoint=checkpoint_path)
    else:
        model, processor, global_step, start_epoch = initialize_model(config, checkpoint=None)
    
    completed_steps = 0 if not resume else global_step
    logd("✅ Done.")

    if config.optimizer.name == "AdamW":
        optimizer = optim.AdamW(
            list(model.parameters()),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=tuple(config.optimizer.betas),
            eps=config.optimizer.eps
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")
    
    if resume:
        checkpoint_data = torch.load(checkpoint_path, map_location=config.device)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        logd(f"Resumed optimizer state from global_step {global_step}")

    num_epochs = config.train.epochs
    steps_per_epoch = train_length
    total_training_steps = num_epochs * steps_per_epoch
    if config.scheduler.name == "cosine":
        # TODO: Switch to a scheduler that combines warmup and cosine decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_training_steps,
            eta_min=config.optimizer.lr * config.scheduler.min_lr_ratio
        )
        if hasattr(config.scheduler, 'max_steps') and config.scheduler.max_steps != total_training_steps:
            logd(f"  ⚠️  Note: config.scheduler.max_steps ({config.scheduler.max_steps}) "
                 f"overridden with calculated value ({total_training_steps})")
        
        scheduler.last_epoch = completed_steps - 1 if completed_steps > 0 else -1

    
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary_writer = SummaryWriter(log_dir=output_dirs['tensorboard'])
    eval_frequency = config.train.eval_frequency

    logd("Starting training.")
    for epoch in range(start_epoch, num_epochs):
        model.train()

        desc_str = f"Epoch {epoch+1}/{num_epochs}"
        batch_bar = tqdm(
            train_loader, desc=desc_str, total=train_length) if config.logging.do_tqdm else train_loader

        for batch in batch_bar:
            captions = batch["caption"]
            images = batch["image"]
            inputs = processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
            )

            optimizer.zero_grad()
            outputs = model(input_ids=inputs.input_ids.to(device),
                            attention_mask=inputs.attention_mask.to(device),
                            pixel_values=inputs.pixel_values.to(device),
                            return_loss=True)

            # Optionally extract embeddings
            # text_embeddings = outputs.text_embeds
            # image_embeddings = outputs.image_embeds

            loss = outputs.loss
            if loss > 0.0 and config.logging.do_tqdm:
                batch_bar.set_postfix(t_loss=f"{loss.item():.4f}")
            summary_writer.add_scalar("Loss/train", loss.item(), global_step)

            loss.backward()
            optimizer.step()
            global_step += 1
            scheduler.step()

        # Eval
        if (epoch + 1) % eval_frequency == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                num_batches = 0
                for val_batch in val_loader:
                    num_batches += 1
                    captions = val_batch["caption"]
                    images = val_batch["image"]
                    inputs = processor(
                        text=captions,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                    )
                    outputs = model(
                        input_ids=inputs.input_ids.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                        pixel_values=inputs.pixel_values.to(device),
                        return_loss=True
                    )
                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / num_batches
            summary_writer.add_scalar("Loss/val", avg_val_loss, epoch)
        # Checkpoint
        save_freq = config.train.save_frequency
        if save_freq != -1 and (epoch + 1) % save_freq == 0:
            checkpoint_path = f"{output_dirs['checkpoints']}/epoch_{epoch+1}.pt"
            save_model_checkpoint(model, optimizer, epoch,
                                  global_step, checkpoint_path)
    summary_writer.close()


def read_config(config_path):
    from train.config_schema import read_config
    return read_config(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a specific checkpoint to resume from')
    args = parser.parse_args()

    config_path = args.config
    config = read_config(config_path)
    train(config, resume=args.resume, checkpoint=args.checkpoint)
