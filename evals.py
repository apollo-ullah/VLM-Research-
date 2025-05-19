import os
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import subprocess
import re
import argparse

# Try to import sentence_transformers, but don't fail if not available
SBERT_AVAILABLE = False
try:
    # First try to check transformers version
    import transformers
    transformers_version = transformers.__version__
    logger = logging.getLogger(__name__)
    logger.info(f"Current transformers version: {transformers_version}")
    
    # Attempt to import sentence_transformers
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    logger.info("Successfully imported sentence_transformers")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {str(e)}")
    logger.warning("sentence_transformers package not available or incompatible.")
    logger.warning("To enable semantic similarity, try installing an older version:")
    logger.warning("pip install sentence-transformers==2.0.0")
    logger.warning("If that doesn't work, try other versions: 1.2.0, 1.1.0")

# Set up logging - reduce verbosity
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("punkt", quiet=True)

# Global variable to control verbosity
VERBOSE = False  # Set to False for minimal output

class TestDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor=None, max_samples=None):
        self.image_dir = Path(image_dir)
        self.processor = processor

        # Load captions
        logger.info(f"Loading test captions from {captions_file}")
        with open(captions_file, "r") as f:
            self.captions_data = json.load(f)
        
        logger.info(f"Loaded {len(self.captions_data)} caption entries from file")
        
        # Process captions in industry-standard format (multiple captions per image)
        self.valid_images = []
        
        # Check if data follows the new format (list of dicts with "captions" list)
        if len(self.captions_data) > 0 and "captions" in self.captions_data[0]:
            logger.info("Detected industry-standard format with multiple captions per image")
            
            for item in self.captions_data:
                image_path = Path(item["image_path"])
                if image_path.exists():
                    try:
                        # Verify the image is readable
                        with Image.open(image_path) as img:
                            width, height = img.size
                        
                        # Add to valid images
                        self.valid_images.append({
                            "image_path": str(image_path),
                            "image_id": item.get("image_id", image_path.stem),
                            "captions": item["captions"]
                        })
                    except Exception as e:
                        logger.error(f"Error verifying image {image_path}: {e}")
                        continue
        else:
            # Legacy format handling
            # Try to convert to new format by grouping captions by image_id
            image_captions = {}
            
            # Handle different JSON formats
            if isinstance(self.captions_data, list):
                # Format: List of dicts with image_id and caption
                for item in self.captions_data:
                    if "image_id" in item and "caption" in item:
                        image_id = str(item["image_id"])
                        if image_id not in image_captions:
                            image_captions[image_id] = {
                                "image_id": image_id,
                                "captions": []
                            }
                        image_captions[image_id]["captions"].append(item["caption"])
                        
            elif (
                isinstance(self.captions_data, dict) and "annotations" in self.captions_data
            ):
                # COCO format: Dict with 'annotations' key
                for anno in self.captions_data["annotations"]:
                    if "image_id" in anno and "caption" in anno:
                        image_id = str(anno["image_id"])
                        if image_id not in image_captions:
                            image_captions[image_id] = {
                                "image_id": image_id,
                                "captions": []
                            }
                        image_captions[image_id]["captions"].append(anno["caption"])
            
            # Create mapping of image IDs to filenames
            id_to_filename = {}
            for filename in self.image_dir.glob("*.jpg"):
                try:
                    # Try to extract image ID from filename
                    image_id = str(int(filename.stem.split("_")[-1]))
                    id_to_filename[image_id] = filename.name
                except (ValueError, IndexError):
                    # If we can't parse the ID, use the filename as is
                    image_id = filename.stem
                    id_to_filename[image_id] = filename.name
            
            # Create list of valid image-caption pairs
            for image_id, item in image_captions.items():
                if image_id in id_to_filename:
                    image_path = self.image_dir / id_to_filename[image_id]
                    if image_path.exists():
                        try:
                            # Verify image is readable
                            with Image.open(image_path) as img:
                                width, height = img.size
                            
                            # Add to valid images
                            item["image_path"] = str(image_path)
                            self.valid_images.append(item)
                        except Exception as e:
                            logger.error(f"Error verifying image {image_path}: {e}")
                            continue

        # Optional: Limit the number of samples if specified (but not by default)
        if max_samples and max_samples < len(self.valid_images):
            self.valid_images = self.valid_images[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")
        else:
            logger.info(f"Using full test set with {len(self.valid_images)} images")

        logger.info(f"Loaded {len(self.valid_images)} valid test images")
        
        # Calculate total captions
        total_captions = sum(len(item["captions"]) for item in self.valid_images)
        logger.info(f"Total of {total_captions} captions, average {total_captions/len(self.valid_images):.2f} per image")

        # Print some sample pairs for verification (only if verbose)
        if VERBOSE and self.valid_images:
            logger.info("Sample entries:")
            for i, item in enumerate(self.valid_images[:3]):
                logger.info(f"\nSample {i + 1}:")
                logger.info(f"Image path: {item['image_path']}")
                logger.info(f"Image ID: {item['image_id']}")
                logger.info(f"Number of captions: {len(item['captions'])}")
                logger.info(f"Sample captions:")
                for j, caption in enumerate(item['captions'][:3]):  # Show up to 3 captions
                    logger.info(f"  {j+1}. {caption}")
                # Try to get image dimensions
                try:
                    with Image.open(item["image_path"]) as img:
                        width, height = img.size
                        logger.info(f"Image dimensions: {width}x{height}")
                except Exception as e:
                    logger.error(f"Error getting image dimensions: {e}")
                logger.info("-" * 30)

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        item = self.valid_images[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            return {
                "image": image,
                "captions": item["captions"],  # Return all captions for evaluation
                "image_path": item["image_path"],
                "image_id": item["image_id"]
            }
        except Exception as e:
            logger.error(f"Error loading image {item['image_path']}: {e}")
            raise


def load_model(model_path, device):
    """Load a BLIP-2 model from path"""
    logger.info(f"Loading model from {model_path}")
    try:
        # Use specific processor for BLIP-2
        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            trust_remote_code=True,
            use_fast=False,  # Use slow tokenizer to avoid compatibility issues
        )

        # First check if this is a PEFT model path
        is_peft_model = False
        try:
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                is_peft_model = True
                logger.info(f"Detected PEFT model at {model_path}")
        except Exception:
            pass

        # Load the base model first
        base_model = AutoModelForVision2Seq.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Move base model to device
        if device.type == "cuda":
            base_model = base_model.to(device)
            
        logger.info("Base model loaded successfully")

        # If it's a PEFT model, load the adapter
        if is_peft_model and model_path != "Salesforce/blip2-opt-2.7b":
            try:
                from peft import PeftModel
                # Load PEFT model
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info(f"Loaded PEFT adapters from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load PEFT model: {e}")
                logger.info("Using base model instead")
                model = base_model
        else:
            model = base_model

        # Verify model device
        model_device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {model_device}")
        
        # Verify model dtype
        model_dtype = next(model.parameters()).dtype
        logger.info(f"Model loaded with dtype: {model_dtype}")

        # Set generation config
        model.config.use_cache = True  # Enable KV cache for faster generation
        
        # Make sure model is in eval mode
        model.eval()

        logger.info("Model and processor loaded successfully")
        return model, processor

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        logger.error("Full error traceback:", exc_info=True)
        raise


def generate_caption(model, processor, image, max_length=20):
    """Generate caption for a single image"""
    try:
        # Ensure image is in the correct format
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Process image
        inputs = processor(
            images=image, return_tensors="pt", padding=True, truncation=True
        )

        # Get model device and dtype
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        # Move inputs to the same device as model
        inputs = {
            k: (
                v.to(device=model_device, dtype=model_dtype if v.dtype.is_floating_point else v.dtype)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in inputs.items()
        }

        logger.debug(f"Input device: {inputs['pixel_values'].device}, dtype: {inputs['pixel_values'].dtype}")
        
        # Generate caption with simplified generation config
        with torch.no_grad():
            try:
                # Use a simpler generation with fewer parameters to reduce errors
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    max_length=20,
                    num_beams=3,
                    repetition_penalty=1.3,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                
                if generated_ids is None or len(generated_ids) == 0:
                    logger.warning("Model returned empty generated_ids")
                    return ""
                    
            except Exception as gen_error:
                logger.error(f"Error during generation: {gen_error}")
                logger.error(f"Model device: {model_device}, dtype: {model_dtype}")
                return ""

        # Decode the generated ids
        try:
            # First check that generated_ids has valid content
            if generated_ids.numel() == 0:
                logger.warning("Generated IDs is empty")
                return ""
                
            # Use processor's batch_decode
            caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Additional check for valid caption
            if not caption or not isinstance(caption[0], str):
                logger.warning(f"Invalid caption result: {type(caption)}")
                return ""
                
            # Clean the caption
            caption_text = caption[0].strip()
            caption_text = clean_caption(caption_text)
            return caption_text
            
        except Exception as decode_error:
            logger.error(f"Error decoding caption: {decode_error}")
            return ""

    except Exception as e:
        logger.error(f"Error in generate_caption: {e}")
        return ""


def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images in batches"""
    # Separate images and other data
    images = [item["image"] for item in batch]
    captions = [item["captions"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    # Return as a dictionary
    return {
        "image": images,  # List of PIL Images
        "captions": captions,  # List of lists of captions
        "image_path": image_paths,  # List of image paths
    }


# Initialize SBERT model
def load_sbert_model():
    """Load Sentence-BERT model for semantic similarity evaluation"""
    if not SBERT_AVAILABLE:
        return None
        
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SBERT model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SBERT model: {e}")
        return None


def compute_sentence_similarity(sbert_model, reference_captions, generated_caption):
    """Compute semantic similarity between generated caption and references using SBERT"""
    try:
        if not generated_caption or not reference_captions:
            return 0.0
            
        # Get embeddings
        reference_embeddings = sbert_model.encode(reference_captions, convert_to_tensor=True)
        generated_embedding = sbert_model.encode([generated_caption], convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = torch.nn.functional.cosine_similarity(
            generated_embedding, reference_embeddings, dim=1)
        
        # Return maximum similarity score
        return torch.max(cosine_scores).item()
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


def evaluate_model(model, processor, test_loader, device):
    """Evaluate model on test set and return various metrics using multiple references per image"""
    model.eval()
    
    # Load SBERT model for semantic similarity
    sbert_model = load_sbert_model()
    
    # For storing results
    results = []
    
    # For COCO-style evaluation
    gts = {}  # Ground truths: {image_id: [caption1, caption2, ...]}
    res = {}  # Results: {image_id: [generated_caption]}

    # Add metrics for repetition and length
    total_repetitions = 0
    total_length_diff = 0
    total_semantic_sim = 0
    successful_generations = 0

    # Create output directory for images
    output_dir = Path("evaluation_images")
    output_dir.mkdir(exist_ok=True)

    logger.info("\nGenerating and comparing captions...")

    sample_counter = 0
    # Save visualization for first 20 samples only
    max_visualizations = 20
    visualizations_created = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = batch["image"]
        all_references = batch["captions"]  # Now this is a list of lists
        image_paths = batch["image_path"]
        
        if "image_id" in batch:
            image_ids = batch["image_id"]
        else:
            # Create numeric image IDs if not provided
            image_ids = [f"img_{sample_counter + i}" for i in range(len(images))]

        for img_idx, (image, references, img_path, image_id) in enumerate(
            zip(images, all_references, image_paths, image_ids)
        ):
            try:
                # Generate caption
                caption = generate_caption(model, processor, image)
                
                # Skip images with empty captions
                if not caption or not isinstance(caption, str):
                    caption = ""
                    if VERBOSE:
                        logger.warning(f"Empty or invalid caption generated for {img_path}")
                
                # Store in COCO format for evaluation
                if image_id not in gts:
                    gts[image_id] = []
                for ref in references:
                    gts[image_id].append(ref)
                res[image_id] = [caption]
                
                # Store complete result
                result_item = {
                    "image_id": image_id,
                    "image_path": img_path,
                    "references": references,
                    "generation": caption
                }
                results.append(result_item)

                sample_counter += 1

                # Calculate metrics only if we have a valid caption
                if caption:
                    successful_generations += 1
                    
                    # Tokenize for BLEU calculation
                    ref_tokenized = [word_tokenize(ref.lower()) for ref in references]
                    hyp_tokenized = word_tokenize(caption.lower())
                    
                    # Count repetitions in generated caption
                    word_counts = {}
                    for word in hyp_tokenized:
                        word_counts[word] = word_counts.get(word, 0) + 1
                    repetitions = sum(count - 1 for count in word_counts.values())
                    total_repetitions += repetitions

                    # Calculate length difference (using closest reference)
                    ref_lengths = [len(tokens) for tokens in ref_tokenized]
                    hyp_length = len(hyp_tokenized)
                    length_diff = min(abs(hyp_length - ref_len) for ref_len in ref_lengths)
                    total_length_diff += length_diff
                    
                    # Calculate BLEU score using multiple references (industry standard)
                    bleu_score = sentence_bleu(ref_tokenized, hyp_tokenized, 
                                        smoothing_function=SmoothingFunction().method1)
                    
                    # Calculate semantic similarity if SBERT is available
                    if sbert_model:
                        semantic_sim = compute_sentence_similarity(sbert_model, references, caption)
                        total_semantic_sim += semantic_sim
                    else:
                        semantic_sim = 0.0
                else:
                    repetitions = 0
                    # Use average reference length for length difference when no caption
                    ref_lengths = [len(word_tokenize(ref.lower())) for ref in references]
                    length_diff = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
                    bleu_score = 0.0
                    semantic_sim = 0.0

                # Create visualizations for a subset of samples only
                if visualizations_created < max_visualizations:
                    # Create a figure with the image and caption
                    fig = plt.figure(figsize=(10, 12))

                    # Add image
                    plt.subplot(2, 1, 1)
                    plt.imshow(image)
                    plt.axis("off")

                    # Add caption text
                    plt.subplot(2, 1, 2)
                    plt.axis("off")
                    
                    # Format reference captions for display
                    ref_display = "\n".join([f"Ref {i+1}: {ref}" for i, ref in enumerate(references)])
                    
                    caption_text = (
                        f"References:\n{ref_display}\n\n"
                        f"Generated: {caption}\n"
                        f"BLEU Score: {bleu_score:.4f}\n"
                        f"Semantic Sim: {semantic_sim:.4f}\n"
                        f"Repetitions: {repetitions}\n"
                        f"Length diff: {length_diff}"
                    )
                    plt.text(
                        0.5,
                        0.5,
                        caption_text,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        wrap=True,
                    )

                    # Save the figure
                    sample_num = batch_idx * len(images) + img_idx + 1
                    output_path = output_dir / f"sample_{visualizations_created+1}.png"
                    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.5)
                    plt.close(fig)
                    
                    visualizations_created += 1

                    # Log to wandb
                    wandb.log(
                        {
                            f"sample_{visualizations_created}": wandb.Image(
                                str(output_path),
                                caption=f"Generated: {caption}",
                            )
                        }
                    )
                    
                    # Log to console (minimal in non-verbose mode)
                    if VERBOSE:
                        logger.info(f"\nSample {visualizations_created}:")
                        logger.info(f"Image: {img_path}")
                        logger.info(f"References:")
                        for i, ref in enumerate(references):
                            logger.info(f"  {i+1}. {ref}")
                        logger.info(f"Generated: {caption}")
                        logger.info(f"Metrics:")
                        logger.info(f"  - BLEU Score: {bleu_score:.4f}")
                        logger.info(f"  - Semantic Similarity: {semantic_sim:.4f}")
                        logger.info(f"  - Repetitions: {repetitions}")
                        logger.info(f"  - Length difference: {length_diff}")
                        logger.info(f"  - Saved visualization to: {output_path}")
                        logger.info("-" * 30)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_counter}: {e}")
                # Continue with next sample rather than crashing
                continue

    # Log overall statistics
    logger.info(f"Successfully generated captions for {successful_generations}/{sample_counter} images")
    
    # Prepare COCO format for metrics evaluation
    coco_gts = {}
    coco_res = {}

    # Convert to COCO format for evaluation metrics
    for image_id, captions in gts.items():
        coco_gts[image_id] = [{"caption": c} for c in captions if isinstance(c, str) and c.strip()]
    
    for image_id, captions in res.items():
        valid_captions = [c for c in captions if isinstance(c, str) and c.strip()]
        if valid_captions:
            coco_res[image_id] = [{"caption": c} for c in valid_captions]
        else:
            # Add an empty caption to avoid errors
            coco_res[image_id] = [{"caption": ""}]

    # Calculate average metrics for successful generations
    if successful_generations > 0:
        avg_repetitions = total_repetitions / successful_generations
        avg_length_diff = total_length_diff / successful_generations
        avg_semantic_sim = total_semantic_sim / successful_generations if sbert_model else 0.0
    else:
        avg_repetitions = 0
        avg_length_diff = 0
        avg_semantic_sim = 0

    # Calculate corpus-level BLEU score from results
    all_bleu_scores = []
    for item in results:
        if item["generation"]:
            ref_tokenized = [word_tokenize(ref.lower()) for ref in item["references"]]
            hyp_tokenized = word_tokenize(item["generation"].lower())
            try:
                bleu = sentence_bleu(ref_tokenized, hyp_tokenized, 
                                smoothing_function=SmoothingFunction().method1)
                all_bleu_scores.append(bleu)
            except Exception as e:
                if VERBOSE:
                    logger.error(f"Error calculating BLEU: {e}")

    # Compute metrics
    metrics = {
        "avg_bleu": np.mean(all_bleu_scores) if all_bleu_scores else 0,
        "bleu_std": np.std(all_bleu_scores) if all_bleu_scores else 0,
        "avg_semantic_sim": avg_semantic_sim,
        "avg_repetitions": avg_repetitions,
        "avg_length_diff": avg_length_diff,
        "success_rate": successful_generations / sample_counter if sample_counter > 0 else 0,
    }

    # Skip metrics calculation if we don't have enough valid captions
    if len(coco_gts) < 2 or len(coco_res) < 2:
        logger.warning(f"Not enough valid captions for metrics calculation. Valid GT: {len(coco_gts)}, Valid Res: {len(coco_res)}")
        metrics.update({
            "cider": 0.0,
            "spice": 0.0,
            "meteor": 0.0,
            "rouge": 0.0,
        })
        return metrics

    # Calculate COCO evaluation metrics
    logger.info("Calculating additional metrics...")

    # CIDEr
    try:
        cider_scorer = Cider()
        # Convert the format to what CIDEr expects
        cider_gts = {image_id: [item["caption"] for item in captions] 
                    for image_id, captions in coco_gts.items()}
        cider_res = {image_id: [item["caption"] for item in captions]
                    for image_id, captions in coco_res.items()}
        # Check that we have at least one valid caption
        if not any(cider_gts.values()) or not any(cider_res.values()):
            logger.warning("No valid captions for CIDEr calculation")
            metrics["cider"] = 0.0
        else:
            cider_score, _ = cider_scorer.compute_score(cider_gts, cider_res)
            metrics["cider"] = cider_score
            logger.info(f"CIDEr Score: {cider_score:.4f}")
    except Exception as e:
        logger.error(f"Error calculating CIDEr: {e}")
        metrics["cider"] = 0.0

    # SPICE (requires Java)
    try:
        # Check if Java is available
        java_check = subprocess.run(["which", "java"], capture_output=True, text=True)
        if java_check.returncode == 0:
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(cider_gts, cider_res)
            metrics["spice"] = spice_score
            logger.info(f"SPICE Score: {spice_score:.4f}")
        else:
            if VERBOSE:
                logger.warning("Java not found, skipping SPICE metric")
            metrics["spice"] = 0.0
    except Exception as e:
        logger.error(f"Error calculating SPICE: {e}")
        metrics["spice"] = 0.0

    # METEOR
    try:
        # Check if Java is available
        java_check = subprocess.run(["which", "java"], capture_output=True, text=True)
        if java_check.returncode == 0:
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(cider_gts, cider_res)
            metrics["meteor"] = meteor_score
            logger.info(f"METEOR Score: {meteor_score:.4f}")
        else:
            if VERBOSE:
                logger.warning("Java not found, skipping METEOR metric")
            metrics["meteor"] = 0.0
    except Exception as e:
        logger.error(f"Error calculating METEOR: {e}")
        metrics["meteor"] = 0.0

    # ROUGE-L
    try:
        rouge_scorer = Rouge()
        # Convert the format to what ROUGE expects
        rouge_gts = {image_id: [item["caption"] for item in captions] 
                    for image_id, captions in coco_gts.items()}
        rouge_res = {image_id: [item["caption"] for item in captions]
                    for image_id, captions in coco_res.items()}
        # Check that we have at least one valid caption
        if not any(rouge_gts.values()) or not any(rouge_res.values()):
            logger.warning("No valid captions for ROUGE calculation")
            metrics["rouge"] = 0.0
        else:
            rouge_score, _ = rouge_scorer.compute_score(rouge_gts, rouge_res)
            metrics["rouge"] = rouge_score
            logger.info(f"ROUGE-L Score: {rouge_score:.4f}")
    except Exception as e:
        logger.error(f"Error calculating ROUGE-L: {e}")
        metrics["rouge"] = 0.0

    logger.info(f"Saved {visualizations_created} visualizations to {output_dir}")

    return metrics


def clean_caption(caption):
    """Clean up generated captions"""
    if not caption:
        return ""
        
    # Remove HTML/markup artifacts
    caption = re.sub(r"<[^>]*>", "", caption)  # Remove HTML tags
    caption = re.sub(r"\[.*?\]", "", caption)  # Remove markdown links
    
    # Remove suspicious tokens that might be code artifacts
    suspicious_tokens = [
        "strutConnector", "attRot", "guiName", "guiActive", 
        "class=", "target=", "_blank", "rel=", "href=",
        "\\", "//", "/*", "*/", "===", "---"
    ]
    for token in suspicious_tokens:
        caption = caption.replace(token, "")
    
    # Remove repeated sentences
    sentences = caption.split(".")
    unique_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent and sent not in unique_sentences:
            unique_sentences.append(sent)

    # Join back into a single caption
    cleaned = ". ".join(unique_sentences)
    if cleaned and not cleaned.endswith("."):
        cleaned += "."

    # Remove any remaining artifacts
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace(" .", ".")

    # Remove repetitive numerical sequences (like "4090 4090 4090")
    cleaned = re.sub(r"\b(\d+)(\s+\1)+\b", r"\1", cleaned)

    # Remove any standalone numbers that might be artifacts
    cleaned = re.sub(r"\s\d+\s", " ", cleaned)
    
    # Fix common artifacts in generated text
    cleaned = re.sub(r"\b(able|ore|hen|ere)\b\s*$", "", cleaned)  # Remove partial words at end
    cleaned = re.sub(r"^\s*(able|ore|hen|ere)\b", "", cleaned)    # Remove partial words at start
    
    # Remove sentences that are too short (likely incomplete)
    final_sentences = []
    for sentence in cleaned.split('.'):
        if len(sentence.split()) > 2:  # Only keep sentences with more than 2 words
            final_sentences.append(sentence)
            
    cleaned = ". ".join(final_sentences)
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
        
    cleaned = cleaned.strip()
    return cleaned


def main():
    """Main function to evaluate models"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate BLIP-2 models on test data")
    parser.add_argument("--model", choices=["baseline", "full_finetuned", "distilled_finetuned", "all"], 
                        default="all", help="Model to evaluate (default: all)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation (default: cuda if available, else cpu)")
    parser.add_argument("--output_file", default="evaluation_results.json",
                        help="File to save evaluation results to (default: evaluation_results.json)")
    args = parser.parse_args()

    # Initialize device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(project="blip2-evaluation", name=f"model-evaluation-{args.model}")

    # Load dataset paths
    with open("dataset_paths.json", "r") as f:
        paths = json.load(f)

    # Create test dataset with all samples for a comprehensive evaluation
    test_dataset = TestDataset(
        image_dir=paths.get("test_images_dir"),
        captions_file=paths.get("test_captions_file"),
        processor=None,
        max_samples=None,  # Set to None to use the full test set (industry standard)
    )

    # Create test dataloader with appropriate batch size
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,  # Reasonable batch size for evaluation
        shuffle=False,
        num_workers=4,  # Increased workers for faster processing
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Model paths - update to point to the PEFT model directories
    model_paths = {
        "baseline": "Salesforce/blip2-opt-2.7b",
        "full_finetuned": "./blip2-finetuned",  # Directory containing the PEFT model
        "distilled_finetuned": "./blip2-finetuned-distilled",  # Directory containing the PEFT model
    }

    # Load existing results if available
    results = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r") as f:
                results = json.load(f)
            logger.info(f"Loaded existing evaluation results from {args.output_file}")
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}. Starting with empty results.")

    # Determine which models to evaluate
    models_to_evaluate = []
    if args.model == "all":
        models_to_evaluate = list(model_paths.keys())
    else:
        models_to_evaluate = [args.model]

    # Evaluate each selected model
    for model_name in models_to_evaluate:
        if model_name not in model_paths:
            logger.warning(f"Model {model_name} not found in model_paths. Skipping.")
            continue
            
        model_path = model_paths[model_name]
        logger.info(f"\nEvaluating {model_name} model...")

        # Load model
        model, processor = load_model(model_path, device)

        # Update dataset processor
        test_dataset.processor = processor

        # Evaluate
        metrics = evaluate_model(model, processor, test_loader, device)

        # Store results
        results[model_name] = metrics

        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log(
                {
                    f"{model_name}_avg_bleu": metrics["avg_bleu"],
                    f"{model_name}_bleu_std": metrics["bleu_std"],
                    f"{model_name}_avg_semantic_sim": metrics.get("avg_semantic_sim", 0.0),
                    f"{model_name}_avg_repetitions": metrics["avg_repetitions"],
                    f"{model_name}_avg_length_diff": metrics["avg_length_diff"],
                    f"{model_name}_cider": metrics.get("cider", 0.0),
                    f"{model_name}_spice": metrics.get("spice", 0.0),
                    f"{model_name}_meteor": metrics.get("meteor", 0.0),
                    f"{model_name}_rouge": metrics.get("rouge", 0.0),
                }
            )

        # Print results
        logger.info(f"{model_name} Results:")
        logger.info(f"Average BLEU Score: {metrics['avg_bleu']:.4f}")
        logger.info(f"Semantic Similarity: {metrics.get('avg_semantic_sim', 0.0):.4f}")
        logger.info(f"CIDEr Score: {metrics.get('cider', 0.0):.4f}")
        logger.info(f"METEOR Score: {metrics.get('meteor', 0.0):.4f}")
        logger.info(f"ROUGE-L Score: {metrics.get('rouge', 0.0):.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save all results
    with open(args.output_file, "w") as f:
        json.dump(
            {
                k: {
                    "avg_bleu": float(v["avg_bleu"]),
                    "bleu_std": float(v["bleu_std"]),
                    "avg_semantic_sim": float(v.get("avg_semantic_sim", 0.0)),
                    "avg_repetitions": float(v["avg_repetitions"]),
                    "avg_length_diff": float(v["avg_length_diff"]),
                    "cider": float(v.get("cider", 0.0)),
                    "spice": float(v.get("spice", 0.0)),
                    "meteor": float(v.get("meteor", 0.0)),
                    "rouge": float(v.get("rouge", 0.0)),
                }
                for k, v in results.items()
            },
            f,
            indent=2,
        )

    if not args.no_wandb:
        wandb.finish()
    logger.info("Evaluation complete! Results saved to " + args.output_file)

    # Print comparison if we evaluated more than one model
    if len(results) > 1:
        logger.info("\nModel Comparison:")
        logger.info("-" * 50)
        for model_name, result in results.items():
            logger.info(f"{model_name}:")
            logger.info(f"  BLEU: {result['avg_bleu']:.4f}, Semantic: {result.get('avg_semantic_sim', 0.0):.4f}")
            logger.info(f"  CIDEr: {result.get('cider', 0.0):.4f}, METEOR: {result.get('meteor', 0.0):.4f}, ROUGE: {result.get('rouge', 0.0):.4f}")

if __name__ == "__main__":
    main()
