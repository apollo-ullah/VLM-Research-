# Enhanced VLM Distillation Script with Text Cleaning and VLM-based Quality Assessment
# Uses LLAVA v1.1.1 for direct assessment of image-caption relevance for VLM training

import os
import sys
import json
import random
import numpy as np
import re
import unicodedata
from PIL import Image
import torch
import traceback
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentError

# Add LLaVA to Python path
LLAVA_DIR = "/workspace/LLaVA"
if os.path.exists(LLAVA_DIR) and LLAVA_DIR not in sys.path:
    sys.path.insert(0, LLAVA_DIR)
    print(f"Added {LLAVA_DIR} to Python path")

# Try to import LLaVA (using transformers)
LLAVA_AVAILABLE = False
try:
    # We just need to check if transformers is installed with proper classes
    import transformers
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    LLAVA_AVAILABLE = True
    print("✓ VLM packages found, will use VLM for quality assessment")
except ImportError as e:
    print(f"⚠️ VLM import error: {e}")
    print("Will proceed without VLM-based assessment.")

# Define model IDs
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
QUALITY_MODEL_ID = "distilroberta-base"  # For caption quality assessment
# For VLM assessment - using the HF-native LLaVA port
VLM_MODEL_ID = "llava-hf/llava-1.6-7b-hf"  # Latest public LLaVA model

# Force GPU usage and print CUDA information
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.set_device(0)
    print(f"GPU Enabled: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(
        f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    DEVICE = "cpu"
    print("WARNING: GPU not available, falling back to CPU. This will be slow!")

# Parse command line arguments for dataset percentage
DATASET_PERCENTAGE = 0.01  # Default is 1% of the dataset (as decimal)

# Simple argument parsing that ignores Jupyter's flags
try:
    # Check if any arguments look like percentages
    input_percentage = 1.0  # Default to 1%
    for arg in sys.argv[1:]:
        # Skip arguments that look like Jupyter flags (-f and files)
        if arg.startswith('-f') or arg.endswith('.json'):
            continue
        # Try to parse as float
        try:
            value = float(arg)
            if 0.01 <= value <= 100:
                input_percentage = value
                break
        except ValueError:
            # Not a valid percentage, ignore it
            continue
            
    # Convert from percentage to decimal (e.g., 1% -> 0.01)
    DATASET_PERCENTAGE = input_percentage / 100
    print(f"Using {input_percentage}% of the dataset")
except Exception as e:
    # If anything goes wrong, use the default
    print(f"Warning: {e}. Using default 1% of dataset.")
    DATASET_PERCENTAGE = 0.01

# Load dataset paths
print("Loading dataset paths...")
try:
    with open("dataset_paths.json", "r") as f:
        paths = json.load(f)

    # Use train split by default for distillation
    IMAGES_DIR = paths.get("train_images_dir")
    CAPTIONS_FILE = paths.get("train_captions_file")
    
    # For backward compatibility
    if not IMAGES_DIR and "images_dir" in paths:
        IMAGES_DIR = paths["images_dir"]
    if not CAPTIONS_FILE and "captions_file" in paths:
        CAPTIONS_FILE = paths["captions_file"]
    
    if not IMAGES_DIR or not CAPTIONS_FILE:
        raise ValueError("Required paths not found in dataset_paths.json")

    print(
        f"Using training split for distillation: Images={IMAGES_DIR}, Captions={CAPTIONS_FILE}"
    )
except (FileNotFoundError, ValueError) as e:
    print(f"⚠️ Error loading dataset_paths.json: {str(e)}")
    print("Please enter paths manually:")
    IMAGES_DIR = input("Enter images directory path: ")
    CAPTIONS_FILE = input("Enter captions file path: ")

# Verify paths exist
print(f"Checking dataset paths...")
if os.path.exists(IMAGES_DIR):
    print(f"✓ Images directory found: {IMAGES_DIR}")
    num_images = len(
        [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    )
    print(f"  Contains {num_images} image files")
else:
    print(f"✗ Images directory not found: {IMAGES_DIR}")

if os.path.exists(CAPTIONS_FILE):
    print(f"✓ Captions file found: {CAPTIONS_FILE}")
    with open(CAPTIONS_FILE, "r") as f:
        captions_data = json.load(f)
    num_entries = len(captions_data)
    print(f"  Contains {num_entries} caption entries")
else:
    print(f"✗ Captions file not found: {CAPTIONS_FILE}")
    num_entries = 0

# Set up output directory
OUTPUT_DIR = "./vlm_distillation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters - adjust based on available memory and dataset size
# For MS COCO dataset (which has ~80K training images with multiple captions each)
# Calculate sample size based on the percentage of the dataset
SAMPLE_SIZE = int(min(90000, num_entries) * DATASET_PERCENTAGE) if num_entries > 0 else int(90000 * DATASET_PERCENTAGE)
TARGET_DATASET_SIZE = SAMPLE_SIZE // 2  # Target size is half of the sample size
TOP_N = TARGET_DATASET_SIZE  # Select this many top pairs

print(f"Using device: {DEVICE}")
print(f"Using {DATASET_PERCENTAGE*100:.2f}% of the dataset")
print(f"Will sample up to {SAMPLE_SIZE} entries and select top {TOP_N} pairs")

# Model IDs
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
# Using BERT-based model fine-tuned for quality assessment
QUALITY_MODEL_ID = "distilroberta-base"  # We'll use this for caption quality assessment


# Text cleaning utilities
def contains_foreign_scripts(text):
    """
    Check if text contains non-Latin scripts like Chinese, Japanese, etc.
    """
    script_counts = {}

    for char in text:
        if char.isalpha():
            try:
                script = unicodedata.name(char).split()[0]
                script_counts[script] = script_counts.get(script, 0) + 1
            except ValueError:
                continue

    # Calculate percentage of non-Latin characters
    total_alpha = sum(script_counts.values())
    if total_alpha == 0:
        return False, {}

    non_latin = sum(
        count
        for script, count in script_counts.items()
        if script not in ["LATIN", "BASIC"]
    )

    non_latin_pct = non_latin / total_alpha if total_alpha > 0 else 0

    return non_latin_pct > 0.05, script_counts


def clean_caption(caption):
    """Clean caption by removing foreign language fragments and repetitive patterns"""
    # List of known foreign text patterns from the logs
    foreign_patterns = [
        r"免版税图像.*?$",  # Chinese stock photo text
        r"免版税.*?$",  # Shortened version
        r"版權商用圖片.*?$",  # Traditional Chinese
        r"版權商用.*?$",  # Shortened version
        r"專業[圖图].*?$",  # Professional image text
        r"新聞類圖片.*?$",  # News category image
        r".*?專業摄影.*?$",  # Professional photography
        r"杂志图[像像].*?$",  # Magazine image
        r"／.*?$",  # Text after forward slash
    ]

    # Apply all patterns
    clean_text = caption
    for pattern in foreign_patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.DOTALL)

    # Remove any isolated non-Latin characters surrounded by spaces
    clean_text = re.sub(
        r"\s[\u4e00-\u9fff\u3040-\u30ff\u0370-\u03ff\u0400-\u04ff]+\s", " ", clean_text
    )

    # Remove repeated sentences (common in the dataset)
    sentences = re.split(r"([.!?])\s+", clean_text)
    # Group sentences with their punctuation
    if len(sentences) % 2 == 1:
        sentences.append("")

    sentence_pairs = [
        (sentences[i], sentences[i + 1]) for i in range(0, len(sentences), 2)
    ]

    # Filter out duplicate sentences
    seen_sentences = set()
    unique_pairs = []

    for sentence, punct in sentence_pairs:
        clean_sentence = sentence.lower().strip()
        if clean_sentence and clean_sentence not in seen_sentences:
            seen_sentences.add(clean_sentence)
            unique_pairs.append((sentence, punct))

    # Rejoin text
    clean_text = ""
    for sentence, punct in unique_pairs:
        clean_text += sentence + punct + " "

    # Cleanup extra whitespace
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Ensure caption ends with proper punctuation
    if clean_text and not clean_text[-1] in [".", "!", "?"]:
        clean_text += "."

    return clean_text


# Image validation utility
def validate_image_file(image_path):
    """
    Validate that an image file exists and can be opened.
    Returns the loaded PIL image if valid, None otherwise.
    """
    if not os.path.exists(image_path):
        print(f"  Warning: Image file not found: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        # Try to load the image data to verify it's valid
        img.load()
        return img
    except Exception as e:
        print(f"  Warning: Failed to load image {image_path}: {e}")
        return None


# Load image-caption pairs with validation
def load_image_caption_pairs(validate_images=True):
    """Load image-caption pairs from the dataset with image validation"""
    print(f"Loading captions from {CAPTIONS_FILE}")
    # Check if file exists
    if not os.path.exists(CAPTIONS_FILE):
        print(f"ERROR: Captions file not found at {CAPTIONS_FILE}")
        return []

    # Load captions file
    with open(CAPTIONS_FILE, "r") as f:
        captions_data = json.load(f)

    # Create image-caption pairs
    pairs = []
    valid_images = 0
    invalid_images = 0

    # First, check if the data is in our new industry-standard format
    is_new_format = len(captions_data) > 0 and isinstance(captions_data[0], dict) and "captions" in captions_data[0]
    
    if is_new_format:
        print("Detected new multi-caption format")
        for item in captions_data:
            image_path = item.get("image_path", "")
            image_id = item.get("image_id", "")
            captions_list = item.get("captions", [])
            
            # Skip if no image path or no captions
            if not image_path or not captions_list:
                continue
                
            # Validate the image if requested
            if validate_images:
                image = validate_image_file(image_path)
                if image is None:
                    invalid_images += 1
                    continue
                valid_images += 1
                
            # Process each caption for this image
            for caption in captions_list:
                # Clean the caption before adding it
                cleaned_caption = clean_caption(caption)
                
                # Skip if cleaning removed everything
                if not cleaned_caption:
                    print(f"Warning: Cleaning removed entire caption: {caption}")
                    continue
                    
                pairs.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "original_caption": caption,
                    "caption": cleaned_caption,
                    "was_cleaned": caption != cleaned_caption,
                })
                
        print(f"Processed {len(pairs)} image-caption pairs from new format")
        
    else:
        # Handle legacy formats
        image_id_to_captions = {}

        # Handle different JSON formats
        if isinstance(captions_data, list):
            # Format: List of dicts with image_id and caption
            for item in captions_data:
                if "image_id" in item and "caption" in item:
                    image_id = item["image_id"]
                    caption = item["caption"]

                    if image_id not in image_id_to_captions:
                        image_id_to_captions[image_id] = []

                    image_id_to_captions[image_id].append(caption)
        elif isinstance(captions_data, dict) and "annotations" in captions_data:
            # COCO format: Dict with 'annotations' key
            annotations = captions_data["annotations"]
            for anno in annotations:
                if "image_id" in anno and "caption" in anno:
                    image_id = anno["image_id"]
                    caption = anno["caption"]

                    if image_id not in image_id_to_captions:
                        image_id_to_captions[image_id] = []

                    image_id_to_captions[image_id].append(caption)
        else:
            print(f"WARNING: Unrecognized captions format, attempting to process anyway")
            
        # Process legacy formats by finding image files
        for image_file in os.listdir(IMAGES_DIR):
            # Skip non-image files
            if not image_file.endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                # Extract the image ID from the filename
                if "COCO" in image_file:
                    image_id_str = image_file.split("_")[-1].split(".")[0]
                    image_id = int(image_id_str)
                else:
                    # Try to extract number from filename
                    image_id_str = "".join(c for c in image_file if c.isdigit())
                    if image_id_str:
                        image_id = int(image_id_str)
                    else:
                        # Fallback: use the filename as string ID
                        image_id = image_file

                # Check if we have captions for this image
                if image_id in image_id_to_captions:
                    image_path = os.path.join(IMAGES_DIR, image_file)

                    # Validate the image if requested
                    if validate_images:
                        image = validate_image_file(image_path)
                        if image is None:
                            invalid_images += 1
                            continue
                        valid_images += 1

                    # Add each caption as a separate pair
                    for caption in image_id_to_captions[image_id]:
                        # Clean the caption
                        cleaned_caption = clean_caption(caption)

                        # Skip if cleaning removed everything
                        if not cleaned_caption:
                            print(f"Warning: Cleaning removed entire caption: {caption}")
                            continue

                        pairs.append(
                            {
                                "image_id": image_id,
                                "image_path": image_path,
                                "original_caption": caption,
                                "caption": cleaned_caption,
                                "was_cleaned": caption != cleaned_caption,
                            }
                        )
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")

    if validate_images:
        print(
            f"Validated {valid_images + invalid_images} images: {valid_images} valid, {invalid_images} invalid"
        )

    print(f"Created {len(pairs)} image-caption pairs")

    # For processing efficiency, randomly sample a subset of pairs
    if len(pairs) > SAMPLE_SIZE:
        pairs = random.sample(pairs, SAMPLE_SIZE)
        print(f"Selected {len(pairs)} random pairs for processing")

    return pairs


# Load models
def load_models():
    """Load CLIP, quality assessment, and VLM models"""
    print("Loading models...")
    models = {}

    try:
        # Load CLIP
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
        models["clip_processor"] = clip_processor
        models["clip_model"] = clip_model
        print("✓ CLIP model loaded successfully")

        # Create a quality assessment tokenizer and model
        print("Loading quality assessment model...")
        quality_tokenizer = AutoTokenizer.from_pretrained(QUALITY_MODEL_ID)
        quality_model = AutoModelForSequenceClassification.from_pretrained(
            QUALITY_MODEL_ID, num_labels=1
        ).to(DEVICE)
        models["quality_tokenizer"] = quality_tokenizer
        models["quality_model"] = quality_model
        print("✓ Quality assessment model loaded successfully")
        
        # Load the VLM model for direct assessment (if LLAVA is available)
        if LLAVA_AVAILABLE:
            print("Loading VLM (LLaVA) from HuggingFace...")
            try:
                # Import the transformers classes for LLaVA
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                
                # Set up dtype for more efficient processing on GPU
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                
                # Load model directly from transformers - much more reliable
                vlm_model = LlavaForConditionalGeneration.from_pretrained(
                    VLM_MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto" if DEVICE == "cuda" else None
                )
                vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
                
                models["vlm_model"] = vlm_model
                models["vlm_processor"] = vlm_processor
                
                print(f"✓ Successfully loaded VLM from {VLM_MODEL_ID}")
                print(f"  Model type: {vlm_model.__class__.__name__}")
                
            except Exception as e:
                print(f"✗ VLM load failed: {e}")
                traceback.print_exc()
                print("Proceeding without VLM assessment")
        else:
            print("✗ LLAVA not available, skipping VLM model loading")

        # Print model size info if using GPU
        if DEVICE == "cuda":
            clip_size_mb = sum(
                p.numel() * p.element_size() for p in clip_model.parameters()
            ) / (1024**2)
            quality_size_mb = sum(
                p.numel() * p.element_size() for p in quality_model.parameters()
            ) / (1024**2)
            print(f"  CLIP model size: {clip_size_mb:.2f} MB")
            print(f"  Quality model size: {quality_size_mb:.2f} MB")
            if "vlm_model" in models:
                vlm_size_mb = sum(
                    p.numel() * p.element_size() for p in models["vlm_model"].parameters()
                ) / (1024**2)
                print(f"  VLM model size: {vlm_size_mb:.2f} MB")
            print(
                f"  GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            )

        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        print(traceback.format_exc())
        return None


# Score image-caption pair with CLIP
def score_with_clip(pair, clip_processor, clip_model):
    """Score image-caption pair using CLIP"""
    # Load and validate image
    image = validate_image_file(pair["image_path"])
    if image is None:
        return 0.0, None

    # Process inputs
    inputs = clip_processor(
        text=[pair["caption"]], images=[image], return_tensors="pt", padding=True
    ).to(DEVICE)

    # Get similarity score
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Normalize embeddings
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            dim=-1, keepdim=True
        )
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(image_embeds, text_embeds)
        score = similarity.item()

    return score, image_embeds.cpu().numpy()


# Assess caption quality
def assess_caption_quality(pair, quality_tokenizer, quality_model):
    """
    Assess the quality of a caption for VLM training
    This function uses prompt engineering to convert a pretrained model
    into a quality assessment system
    """
    # Create a prompt that frames the quality assessment task
    prompt = f"Evaluate if this caption is high quality for VLM training: '{pair['caption']}'"
    prompt += " Is it clear, accurate, ethical, and without repetition or foreign text fragments?"

    # Tokenize input
    inputs = quality_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    ).to(DEVICE)

    # Get model prediction
    with torch.no_grad():
        outputs = quality_model(**inputs)
        # Convert logits to score in 0-1 range
        quality_score = torch.sigmoid(outputs.logits).item()

    # Additional rule-based checks for specific quality issues
    additional_score = 1.0

    # Penalize if caption is too short
    if len(pair["caption"].split()) < 5:
        additional_score *= 0.8

    # Penalize if caption has very long words (likely garbage)
    if any(len(word) > 20 for word in pair["caption"].split()):
        additional_score *= 0.7

    # Penalize highly repetitive captions
    words = pair["caption"].split()
    if len(words) > 0:
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.5:  # High repetition
            additional_score *= 0.6

    # Combine model prediction with rule-based score
    final_score = quality_score * additional_score

    return final_score


def calculate_caption_diversity(captions):
    """
    Calculate diversity score between multiple captions for the same image.
    Returns a score between 0 and 1, where 1 means highly diverse captions.
    """
    if len(captions) <= 1:
        return 1.0  # Single caption is considered maximally diverse

    # Tokenize and get unique words for each caption
    caption_words = [set(c.lower().split()) for c in captions]

    # Calculate Jaccard similarity between all pairs
    similarities = []
    for i in range(len(caption_words)):
        for j in range(i + 1, len(caption_words)):
            intersection = len(caption_words[i] & caption_words[j])
            union = len(caption_words[i] | caption_words[j])
            if union > 0:
                similarities.append(intersection / union)

    # Convert average similarity to diversity score
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    diversity_score = 1 - avg_similarity

    return diversity_score


def select_best_captions_for_image_with_vlm(
    image_captions, clip_scores, quality_scores, vlm_scores, max_captions_per_image=5
):
    """
    Select the best captions for a single image based on multiple criteria including VLM assessment.
    Returns a list of selected caption indices and their scores.
    """
    if not image_captions:
        return []

    # Calculate diversity score for all captions
    diversity_score = calculate_caption_diversity(image_captions)

    # Score each caption
    caption_scores = []
    for i, (caption, clip_score, quality_score, vlm_score) in enumerate(
        zip(image_captions, clip_scores, quality_scores, vlm_scores)
    ):
        # Combine scores with diversity - adjust weights based on availability of VLM
        if LLAVA_AVAILABLE:
            # Standard weights with VLM
            combined_score = (
                0.3 * clip_score            # CLIP score (image-caption alignment)
                + 0.2 * quality_score       # Caption quality
                + 0.4 * vlm_score           # VLM relevance assessment (highest weight)
                + 0.1 * diversity_score     # Caption diversity
            )
        else:
            # Adjusted weights without VLM
            combined_score = (
                0.6 * clip_score            # Increased weight for CLIP without VLM
                + 0.3 * quality_score       # Increased weight for quality assessment
                + 0.1 * diversity_score     # Same diversity weight
            )

        caption_scores.append(
            {
                "index": i,
                "caption": caption,
                "clip_score": clip_score,
                "quality_score": quality_score,
                "vlm_score": vlm_score,
                "diversity_score": diversity_score,
                "combined_score": combined_score,
            }
        )

    # Sort by combined score
    caption_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    # Select top captions, ensuring diversity
    selected_captions = []
    selected_words = set()

    for score_info in caption_scores:
        if len(selected_captions) >= max_captions_per_image:
            break

        # Get words in current caption
        current_words = set(score_info["caption"].lower().split())

        # Calculate overlap with already selected captions
        overlap = (
            len(current_words & selected_words) / len(current_words)
            if current_words
            else 0
        )

        # If overlap is low enough, add this caption
        if overlap < 0.7:  # Allow up to 70% word overlap
            selected_captions.append(score_info)
            selected_words.update(current_words)

    return selected_captions


def score_image_caption_pairs(pairs, models):
    """Score all image-caption pairs using CLIP, quality assessment, and VLM judgment"""
    print("Scoring image-caption pairs...")

    # Unpack models
    clip_processor = models["clip_processor"]
    clip_model = models["clip_model"]
    quality_tokenizer = models["quality_tokenizer"]
    quality_model = models["quality_model"]
    
    # Get VLM models if available
    vlm_processor = models.get("vlm_processor")
    vlm_model = models.get("vlm_model")
    use_vlm = vlm_processor is not None and vlm_model is not None
    
    if use_vlm:
        print("✓ Using VLM for direct assessment of image-caption relevance")
    else:
        print("✗ VLM not available, relying on CLIP and quality assessment only")

    # Group pairs by image_id
    image_groups = {}
    for pair in pairs:
        image_id = pair["image_id"]
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(pair)

    scored_pairs = []

    # Free up GPU memory if possible
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(
            f"GPU memory after cache cleared: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )

    # Process each image group
    for image_id, group_pairs in tqdm(image_groups.items(), desc="Processing images"):
        try:
            # Score all captions for this image
            image_captions = []
            clip_scores = []
            quality_scores = []
            vlm_scores = []
            image_path = group_pairs[0]["image_path"]  # All pairs in group have same image

            for pair in group_pairs:
                # Score with CLIP
                clip_score, _ = score_with_clip(pair, clip_processor, clip_model)

                # Assess caption quality
                quality_score = assess_caption_quality(
                    pair, quality_tokenizer, quality_model
                )
                
                # Get VLM relevance assessment if available
                if use_vlm:
                    vlm_score = assess_vlm_relevance(pair, vlm_processor, vlm_model)
                else:
                    vlm_score = 0.5  # Neutral score if VLM not available
                
                image_captions.append(pair["caption"])
                clip_scores.append(clip_score)
                quality_scores.append(quality_score)
                vlm_scores.append(vlm_score)

            # Select best captions for this image
            selected_captions = select_best_captions_for_image_with_vlm(
                image_captions,
                clip_scores,
                quality_scores,
                vlm_scores,
                max_captions_per_image=5,  # Adjust this parameter as needed
            )

            # Add selected captions to scored pairs
            for score_info in selected_captions:
                original_pair = group_pairs[score_info["index"]]
                scored_pair = original_pair.copy()
                scored_pair.update(
                    {
                        "clip_score": score_info["clip_score"],
                        "quality_score": score_info["quality_score"],
                        "vlm_score": score_info["vlm_score"],
                        "diversity_score": score_info["diversity_score"],
                        "combined_score": score_info["combined_score"],
                    }
                )
                scored_pairs.append(scored_pair)

            # Print progress details
            if len(scored_pairs) % 10 == 0:
                print(f"Processed {len(scored_pairs)} total caption selections")
                print(f"  Last image: {os.path.basename(image_path)}")
                print(
                    f"  Selected {len(selected_captions)} captions from {len(group_pairs)} available"
                )

            # Periodically clear GPU cache
            if DEVICE == "cuda" and len(scored_pairs) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing image group {image_id}: {e}")
            traceback.print_exc()

    return scored_pairs


def select_top_pairs(scored_pairs, top_n=TOP_N):
    """Select top-N pairs based on combined scores"""
    print(f"Selecting top pairs...")

    # Sort by combined score (descending)
    sorted_pairs = sorted(scored_pairs, key=lambda x: x["combined_score"], reverse=True)

    # Select top-N
    top_pairs = sorted_pairs[:top_n]

    # Calculate statistics
    total_images = len(set(p["image_id"] for p in top_pairs))
    total_captions = len(top_pairs)
    avg_captions_per_image = total_captions / total_images if total_images > 0 else 0

    print(
        f"\nSelected {len(top_pairs)} caption pairs from {total_images} unique images"
    )
    print(f"Average captions per image: {avg_captions_per_image:.2f}")

    print("\nTop 5 pairs:")
    for i, pair in enumerate(top_pairs[:5]):
        print(
            f"{i+1}. Image: {os.path.basename(pair['image_path'])}"
            f"\n   Combined score: {pair['combined_score']:.4f}"
            f"\n   CLIP: {pair['clip_score']:.4f}, Quality: {pair['quality_score']:.4f}"
            f"\n   VLM: {pair.get('vlm_score', 0):.4f}, Diversity: {pair.get('diversity_score', 0):.4f}"
            f"\n   Caption: {pair['caption'][:100]}..."
        )

    # Calculate average scores
    avg_clip = sum(p["clip_score"] for p in top_pairs) / len(top_pairs)
    avg_quality = sum(p["quality_score"] for p in top_pairs) / len(top_pairs)
    avg_vlm = sum(p.get("vlm_score", 0) for p in top_pairs) / len(top_pairs)
    avg_diversity = sum(p.get("diversity_score", 0) for p in top_pairs) / len(top_pairs)
    avg_combined = sum(p["combined_score"] for p in top_pairs) / len(top_pairs)

    print("\nSelected pairs statistics:")
    print(f"Average CLIP score: {avg_clip:.4f}")
    print(f"Average Quality score: {avg_quality:.4f}")
    print(f"Average VLM score: {avg_vlm:.4f}")
    print(f"Average Diversity score: {avg_diversity:.4f}")
    print(f"Average Combined score: {avg_combined:.4f}")

    return top_pairs


def save_distilled_dataset(pairs, output_path):
    """Save distilled dataset to file"""
    # Clean pairs for saving (remove embeddings)
    clean_pairs = []
    for pair in pairs:
        clean_pair = {k: v for k, v in pair.items() if k != "image_embedding"}
        clean_pairs.append(clean_pair)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(clean_pairs, f, indent=2)

    print(f"Saved distilled dataset to {output_path}")


# Function to assess image-caption relevance using the VLM
def assess_vlm_relevance(pair, vlm_processor, vlm_model):
    """
    Ask the VLM model if the image-caption pair is relevant for training a VLM
    Returns a relevance score between 0-1
    """
    if vlm_model is None:
        return 0.5  # Default score if VLM is not available
    
    # Load and validate image
    image_path = pair["image_path"]
    image = validate_image_file(image_path)
    if image is None:
        return 0.0
    
    try:
        # Standard industry prompt for relevance assessment
        prompt = (
            "Is this caption relevant and high-quality for this image for training "
            "a vision-language model on image captioning? "
            "Answer only 'yes' or 'no'.\n"
            f'Caption: "{pair["caption"]}"'
        )
        
        # Process inputs - single function call handles both image and text
        inputs = vlm_processor(prompt, images=image, return_tensors="pt").to(DEVICE)
        
        # Generate response with torch.inference_mode() for maximum efficiency
        with torch.inference_mode():
            output = vlm_model.generate(**inputs, max_new_tokens=5)
        
        # Decode output
        answer = vlm_processor.decode(output[0], skip_special_tokens=True).lower()
        
        # Simple scoring based on yes/no detection
        if "yes" in answer and "no" not in answer:
            return 0.9  # High score for yes
        elif "no" in answer and "yes" not in answer:
            return 0.1  # Low score for no
        else:
            return 0.5  # Neutral for ambiguous answers
    
    except Exception as e:
        print(f"Error in VLM assessment: {e}")
        # Return moderate score on error
        return 0.5


# Main pipeline function
def run_enhanced_vlm_pipeline():
    """Run the enhanced VLM data distillation pipeline with VLM-based relevance assessment"""
    # Print initial GPU state
    if DEVICE == "cuda":
        print("\n--- Initial GPU State ---")
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    print("\n=== VLM-Based Dataset Distillation Pipeline ===")
    print("This pipeline uses a teacher VLM (like LLAVA) to assess image-caption relevance")
    print("for training vision-language models on image captioning tasks.")
    print("The approach combines:")
    print(" 1. CLIP-based image-caption alignment scores")
    print(" 2. Text quality assessment")
    print(" 3. Direct VLM judgment of caption relevance")
    print(" 4. Caption diversity measurements")
    print("\nThe result is a high-quality distilled dataset optimized for VLM training.")

    # Load image-caption pairs with validation and caption cleaning
    pairs = load_image_caption_pairs(validate_images=True)
    if not pairs:
        print("No image-caption pairs found. Check your dataset paths.")
        return

    # Report actual percentage of dataset used
    if num_entries > 0:
        actual_percentage = (len(pairs) / num_entries) * 100
        print(f"Actually using {len(pairs)} pairs, which is {actual_percentage:.2f}% of the full dataset ({num_entries} entries)")

    # Load models
    models = load_models()
    if models is None:
        print("Failed to load models. Cannot continue.")
        return

    # Check if VLM model was loaded
    has_vlm = "vlm_model" in models and models["vlm_model"] is not None
    if has_vlm:
        print("\n✓ VLM model successfully loaded and will be used for direct assessment")
    else:
        print("\n⚠️ VLM model not available - proceeding with CLIP and quality assessment only")

    # GPU state after model loading
    if DEVICE == "cuda":
        print("\n--- GPU State After Model Loading ---")
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Score image-caption pairs with CLIP, quality assessment, and VLM
    scored_pairs = score_image_caption_pairs(pairs, models)

    # Select top-N pairs based on combined score
    distilled_pairs = select_top_pairs(scored_pairs)

    # Save distilled dataset
    output_path = os.path.join(OUTPUT_DIR, f"distilled_dataset_{DATASET_PERCENTAGE*100:.0f}pct.json")
    save_distilled_dataset(distilled_pairs, output_path)
    
    # Also save a standardized version for training
    standard_output_path = os.path.join(OUTPUT_DIR, "distilled_dataset.json")
    save_distilled_dataset(distilled_pairs, standard_output_path)

    # Final GPU state
    if DEVICE == "cuda":
        print("\n--- Final GPU State ---")
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        # Clean up
        torch.cuda.empty_cache()
        print(
            f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )

    print("\n=== VLM Distillation Pipeline Complete ===")
    print(f"✓ Selected the top {len(distilled_pairs)} highest-quality image-caption pairs")
    print(f"✓ Output saved to {output_path}")
    print(f"✓ Standard path for training: {standard_output_path}")
    
    if has_vlm:
        avg_vlm_score = sum(p.get("vlm_score", 0) for p in distilled_pairs) / len(distilled_pairs)
        print(f"✓ Average VLM relevance score: {avg_vlm_score:.4f}")
        
    print("\nYou can now use the distilled dataset for fine-tuning your models using distuned.py")
    print("Example: python distuned.py")


# Run the pipeline
if __name__ == "__main__":
    run_enhanced_vlm_pipeline()
