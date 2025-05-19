import subprocess
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_scheduler,
    TrainingArguments,
    Trainer,
)
from PIL import Image
from tqdm import tqdm
import wandb
from peft import get_peft_model, LoraConfig, TaskType
import logging
from pathlib import Path
from dataset import filter_captions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioningDataset(Dataset):
    def __init__(
        self, image_dir, captions_file, processor, max_length=30, is_distilled=False
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_length = max_length
        self.is_distilled = is_distilled

        # Load captions
        logger.info(f"Loading captions from {captions_file}")
        with open(captions_file, "r") as f:
            self.captions_data = json.load(f)
        logger.info(f"Loaded {len(self.captions_data)} caption entries")

        if is_distilled:
            # For distilled dataset, each entry already has image_path and caption
            self.valid_images = []
            for item in self.captions_data:
                image_path = Path(item["image_path"])
                if image_path.exists():
                    self.valid_images.append(
                        {
                            "image_path": str(image_path),
                            "caption": item["caption"],
                            "clip_score": item.get("clip_score", 0),
                            "quality_score": item.get("quality_score", 0),
                            "combined_score": item.get("combined_score", 0),
                        }
                    )
        else:
            # Process captions in industry-standard format (multiple captions per image)
            self.valid_samples = []
            for item in self.captions_data:
                image_path = Path(item["image_path"])
                if not image_path.exists():
                    continue
                
                # Create one training example for each caption (multiple instance learning)
                for caption in item["captions"]:
                    self.valid_samples.append({
                        "image_path": str(image_path),
                        "caption": caption,
                        "image_id": item["image_id"]
                    })

        if is_distilled:
            if not self.valid_images:
                logger.error(f"No valid images found in {self.image_dir}")
                raise ValueError("No valid images found in the specified directory")
            logger.info(f"Loaded {len(self.valid_images)} valid image-caption pairs")
            # Log distillation metrics if available
            if len(self.valid_images) > 0 and all("clip_score" in item for item in self.valid_images):
                logger.info(f"Average CLIP score: {sum(item.get('clip_score', 0) for item in self.valid_images) / len(self.valid_images):.4f}")
                logger.info(f"Average Quality score: {sum(item.get('quality_score', 0) for item in self.valid_images) / len(self.valid_images):.4f}")
                logger.info(f"Average Combined score: {sum(item.get('combined_score', 0) for item in self.valid_images) / len(self.valid_images):.4f}")
        else:
            if not self.valid_samples:
                logger.error(f"No valid samples found in {self.image_dir}")
                raise ValueError("No valid samples found in the specified directory")
            logger.info(f"Loaded {len(self.valid_samples)} valid image-caption pairs")
            
            # Count unique images
            unique_images = len(set(item["image_id"] for item in self.valid_samples))
            logger.info(f"Number of unique images: {unique_images}")
            logger.info(f"Average captions per image: {len(self.valid_samples)/unique_images:.2f}")

        # Print some sample pairs for verification
        if is_distilled and self.valid_images:
            logger.info("Sample pairs:")
            for pair in self.valid_images[:3]:
                logger.info(f"Image: {pair['image_path']}")
                logger.info(f"Caption: {pair['caption']}")
                logger.info("---")
        elif not is_distilled and self.valid_samples:
            logger.info("Sample pairs:")
            for pair in self.valid_samples[:3]:
                logger.info(f"Image: {pair['image_path']}")
                logger.info(f"Caption: {pair['caption']}")
                logger.info("---")

    def __len__(self):
        if self.is_distilled:
            return len(self.valid_images)
        else:
            return len(self.valid_samples)

    def __getitem__(self, idx):
        if self.is_distilled:
            item = self.valid_images[idx]
        else:
            item = self.valid_samples[idx]
            
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {item['image_path']}: {str(e)}")
            raise

        # Get caption
        caption = item["caption"]

        # Process image and text
        try:
            inputs = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
        except Exception as e:
            logger.error(
                f"Error processing image {item['image_path']} with caption: {str(e)}"
            )
            raise

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


def print_model_structure(model, prefix=""):
    """Recursively print model structure to find target modules."""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            logger.info(f"Found potential target module: {full_name}")
        print_model_structure(module, full_name)


def prepare_model_and_processor():
    # Initialize processor and model using Auto classes
    model_name = "Salesforce/blip2-opt-2.7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # Use slow tokenizer to avoid compatibility issues
        )

        # Load base model without PEFT first to inspect its configuration
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Successfully loaded base model and processor")

        # Print model structure to find target modules
        logger.info("Analyzing model structure to find target modules...")
        print_model_structure(base_model)

        # Let's also print the language model structure specifically
        logger.info("\nAnalyzing language model structure...")
        if hasattr(base_model, "language_model"):
            print_model_structure(base_model.language_model, "language_model")

        # Define PEFT configuration with more conservative settings
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,  # Reduced from 8 for stability
            lora_alpha=16,  # Reduced from 32
            lora_dropout=0.05,  # Reduced from 0.1
            # Target only specific projection modules in the language model
            target_modules=["q_proj", "v_proj"],  # Removed k_proj and out_proj
            # Initialize adapter with proper scaling
            init_lora_weights="gaussian",
        )

        # Apply PEFT to the model
        model = get_peft_model(base_model, peft_config)
        model = model.to(device)
        model.print_trainable_parameters()

        return model, processor

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def train_model(
    model,
    train_dataloader,
    eval_dataloader=None,
    num_epochs=3,
    learning_rate=2e-5,
    warmup_steps=200,
    gradient_accumulation_steps=2,
    output_dir="./blip2-finetuned-distilled",
):
    device = model.device
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Initialize wandb
    run_name = os.path.basename(output_dir)
    wandb.init(project="blip2-captioning", name=run_name)

    # Log key hyperparameters
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_dataloader.batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "dataset_size": len(train_dataloader.dataset),
            "scheduler": "cosine",
            "weight_decay": 0.01,
        }
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            try:
                # Prepare model inputs - only include what BLIP-2 expects
                model_inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["input_ids"].to(
                        device
                    ),  # For causal language modeling
                }

                # Forward pass with mixed precision
                with autocast():
                    # Use the base model's forward method directly
                    outputs = model.base_model(**model_inputs)
                    loss = outputs.loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": total_loss / (step + 1)})

                # Log to wandb
                wandb.log(
                    {
                        "loss": loss.item() * gradient_accumulation_steps,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + step / len(train_dataloader),
                    }
                )

            except Exception as e:
                logger.error(f"Error in training step {step}: {str(e)}")
                logger.error(f"Batch keys: {batch.keys()}")
                raise

        # Evaluate if validation data is provided
        if eval_dataloader is not None:
            eval_loss = 0
            model.eval()
            eval_steps = 0

            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
                    try:
                        # Prepare model inputs
                        eval_inputs = {
                            "pixel_values": eval_batch["pixel_values"].to(device),
                            "input_ids": eval_batch["input_ids"].to(device),
                            "attention_mask": eval_batch["attention_mask"].to(device),
                            "labels": eval_batch["input_ids"].to(device),
                        }

                        # Forward pass
                        with autocast():
                            outputs = model.base_model(**eval_inputs)
                            loss = outputs.loss

                        eval_loss += loss.item()
                        eval_steps += 1
                    except Exception as e:
                        logger.error(f"Error in eval step: {str(e)}")
                        continue

            avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0
            logger.info(f"Validation Loss: {avg_eval_loss:.4f}")
            wandb.log({"val_loss": avg_eval_loss, "epoch": epoch + 1})

            # Set back to train mode
            model.train()

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    model.save_pretrained(output_dir)
    logger.info(f"Saved final model to {output_dir}")
    wandb.finish()


def main():
    # Set device and enable mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimize CUDA settings
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("CUDA Optimizations enabled:")
        logger.info("  - TF32 enabled for Ampere")
        logger.info("  - cuDNN benchmark enabled")
        logger.info("  - Deterministic mode disabled")

    # Load dataset paths
    try:
        with open("dataset_paths.json", "r") as f:
            paths = json.load(f)
        logger.info(f"Loaded dataset paths: {paths}")
    except Exception as e:
        logger.error(f"Error loading dataset_paths.json: {str(e)}")
        raise

    # Initialize model and processor
    model, processor = prepare_model_and_processor()

    # Create dataset using distilled data
    try:
        # THE ONLY DIFFERENCE FROM FINETUNE.PY - USING DISTILLED DATASET PATH
        distilled_file = "./vlm_distillation_output/distilled_dataset.json"
        with open(distilled_file, "r") as f:
            distilled_data = json.load(f)
        
        # Apply filtering to clean captions
        filtered_captions = filter_captions(distilled_data)
        
        # Save filtered captions to a new file
        filtered_path = distilled_file.replace(".json", "_filtered.json")
        with open(filtered_path, "w") as f:
            json.dump(filtered_captions, f)
        
        logger.info(f"Saved {len(filtered_captions)} filtered captions to {filtered_path}")
        
        train_dataset = ImageCaptioningDataset(
            image_dir=paths["train_images_dir"],
            captions_file=filtered_path,  # Use filtered captions
            processor=processor,
            is_distilled=True,  # IMPORTANT: Properly set is_distilled flag for distilled dataset
        )
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise

    # Create validation dataset if available
    val_dataset = None
    if "val_captions_file" in paths and "val_images_dir" in paths:
        try:
            val_dataset = ImageCaptioningDataset(
                image_dir=paths["val_images_dir"],
                captions_file=paths["val_captions_file"],
                processor=processor,
                is_distilled=False,
            )
            logger.info(f"Created validation dataset with {len(val_dataset)} examples")
        except Exception as e:
            logger.warning(f"Could not create validation dataset: {e}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Create validation dataloader if validation dataset exists
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Train model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        num_epochs=3,
        learning_rate=2e-5,
        warmup_steps=200,
        gradient_accumulation_steps=2,
        output_dir="./blip2-finetuned-distilled",  # DIFFERENT OUTPUT DIRECTORY FOR DISTILLED MODEL
    )


if __name__ == "__main__":
    main()
