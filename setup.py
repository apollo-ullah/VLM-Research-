#!/usr/bin/env python
# flamingo_peft_train_fixed.py  –  2025‑05‑09
#
# OpenFlamingo + PEFT‑LoRA fine‑tuning, now with a custom
# `resize_embeddings()` that works for the plain Flamingo module.

import os, sys, json, gc, random, logging, traceback
from datetime import datetime
from types import MethodType

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from transformers import PretrainedConfig, get_linear_schedule_with_warmup
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler(f"logs/train_{ts}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seeds set to {seed}")


def monkeypatch_flamingo(model):
    """Give OpenFlamingo just enough HF‑style interface for PEFT."""

    # prepare_inputs_for_generation
    def _prep(self, input_ids=None, **kwargs):
        return {"lang_x": input_ids, **kwargs}

    model.prepare_inputs_for_generation = MethodType(_prep, model)

    # minimal config
    if not hasattr(model, "config"):
        model.config = PretrainedConfig()
    model.config.model_type = "openflamingo"
    model.config.is_encoder_decoder = False

    # input‑embedding accessors
    if not hasattr(model, "get_input_embeddings"):

        def _get(self):
            return self.lang_encoder.get_input_embeddings()

        def _set(self, new_emb):
            self.lang_encoder.set_input_embeddings(new_emb)

        model.get_input_embeddings = MethodType(_get, model)
        model.set_input_embeddings = MethodType(_set, model)
    return model


def resize_embeddings(model, tokenizer):
    """
    Manually grow the language‑side embedding & lm_head to match the
    tokenizer's new vocab size.
    """
    old_emb = model.lang_encoder.get_input_embeddings()
    old_tokens, dim = old_emb.weight.shape
    new_tokens = len(tokenizer)
    if new_tokens <= old_tokens:
        return  # nothing to do

    logger.info(f"Resizing embeddings: {old_tokens} → {new_tokens}")
    new_emb = torch.nn.Embedding(new_tokens, dim)
    new_emb.weight.data[:old_tokens] = old_emb.weight.data
    torch.nn.init.normal_(new_emb.weight.data[old_tokens:], std=0.02)
    model.lang_encoder.set_input_embeddings(new_emb)

    # If the backbone exposes an lm_head tied to embeddings, grow it too
    if hasattr(model.lang_encoder, "lm_head"):
        old_lm = model.lang_encoder.lm_head
        if old_lm.weight.shape[0] == old_tokens:
            new_lm = torch.nn.Linear(dim, new_tokens, bias=False)
            new_lm.weight.data[:old_tokens] = old_lm.weight.data
            torch.nn.init.normal_(new_lm.weight.data[old_tokens:], std=0.02)
            model.lang_encoder.lm_head = new_lm


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class COCODataset(Dataset):
    def __init__(self, json_path, img_dir, proc, tok, max_len=100):
        self.proc, self.tok, self.max_len = proc, tok, max_len
        self.img_dir = img_dir

        # Load COCO format annotations
        with open(json_path) as f:
            data = json.load(f)

        # Handle both COCO format and simplified format
        if isinstance(data, dict) and "annotations" in data:
            # COCO format
            self.annotations = data["annotations"]
            # Group annotations by image_id
            self.img_to_captions = defaultdict(list)
            for ann in self.annotations:
                self.img_to_captions[ann["image_id"]].append(ann["caption"])

            # Create list of (image_id, caption) pairs
            self.samples = []
            for img_id, captions in self.img_to_captions.items():
                # Ensure we have all 5 captions
                if len(captions) < 5:
                    logger.warning(f"Image {img_id} has only {len(captions)} captions")
                for caption in captions:
                    self.samples.append({"image_id": img_id, "caption": caption})
        else:
            # Simplified format
            self.annotations = data
            self.img_to_captions = defaultdict(list)
            for ann in self.annotations:
                self.img_to_captions[ann["image_path"]].append(ann["caption"])

            # Create list of (image_id, caption) pairs
            self.samples = []
            for img_id, captions in self.img_to_captions.items():
                # Ensure we have all 5 captions
                if len(captions) < 5:
                    logger.warning(f"Image {img_id} has only {len(captions)} captions")
                for caption in captions:
                    self.samples.append({"image_id": img_id, "caption": caption})

        # Log dataset statistics
        total_images = len(self.img_to_captions)
        total_captions = len(self.samples)
        avg_captions = total_captions / total_images if total_images > 0 else 0

        logger.info(f"Dataset statistics:")
        logger.info(f"- Total images: {total_images}")
        logger.info(f"- Total captions: {total_captions}")
        logger.info(f"- Average captions per image: {avg_captions:.2f}")

        if avg_captions < 4.5:  # Allow for some missing captions
            logger.warning("Dataset may be incomplete - expected ~5 captions per image")

        if total_images < 18000:  # Allow for some missing images
            logger.warning("Dataset may be incomplete - expected ~18.7K images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        it = self.samples[idx]
        img_id = it["image_id"]

        # Handle both COCO format and simplified format
        if isinstance(img_id, int):
            img_path = os.path.join(self.img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        else:
            img_path = os.path.join(self.img_dir, img_id)

        v = self.proc(Image.open(img_path).convert("RGB"))
        v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)

        prompt = "<image><|endofchunk|>"
        text = f"{prompt} {it['caption']}"
        enc = self.tok(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        labels = enc.input_ids.clone().squeeze(0)
        p_len = len(self.tok(prompt, add_special_tokens=True).input_ids)
        labels[:p_len] = -100
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "vision_x": v,
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
            "image_path": img_path,
            "caption": it["caption"],
            "image_id": img_id,
        }


import os, torch, json, gc
from datetime import datetime
from types import MethodType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------------------------------------------------
# ── helper: make Flamingo accept `input_ids` ───────────────────────
# ------------------------------------------------------------------
# -----------------------------------------------------
# helper ─ patch Flamingo so PEFT is happy
# -----------------------------------------------------
# ---------------------------------------------------------------
# helper ─ make an OpenFlamingo model look like a HF text model
# ---------------------------------------------------------------
from types import MethodType
from transformers import PretrainedConfig

_DISCARD_KWS = {
    "inputs_embeds",
    "position_ids",
    "token_type_ids",
    "past_key_values",
    "use_cache",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}


def patch_flamingo_for_peft(model):
    """Give OpenFlamingo just enough HF‑style interface for PEFT and preserve gradients."""

    # Store original forward
    original_forward = model.forward

    # Define the combined forward patch
    def _forward(self, *args, **kwargs):
        # ---- make PEFT happy -------------------------------------------
        if "input_ids" in kwargs and "lang_x" not in kwargs:
            kwargs["lang_x"] = kwargs.pop("input_ids")

        # Remove unwanted kwargs
        for k in list(kwargs.keys()):
            if k in _DISCARD_KWS:
                kwargs.pop(k)

        # Handle return_loss flag
        return_loss = kwargs.pop("return_loss", False)

        # ---- real forward ----------------------------------------------
        out = original_forward(*args, **kwargs)

        # ---- remove the .detach() --------------------------------------
        if isinstance(out, tuple):
            # Handle tuple output (logits,) or (loss, logits)
            if len(out) == 1:
                logits = out[0]
                loss = None
            else:
                loss, logits = out
        else:
            # Handle FlamingoOutput
            logits = out.logits
            loss = out.loss

        # Return in same format as input
        if not isinstance(out, tuple):
            return type(out)(loss=loss, logits=logits)
        return (loss, logits) if loss is not None else (logits,)

    # Apply the forward patch
    model.forward = MethodType(_forward, model)

    # Add PEFT interface methods
    def _prep(self, input_ids=None, **kwargs):
        return {"lang_x": input_ids, **kwargs}

    model.prepare_inputs_for_generation = MethodType(_prep, model)

    # Add minimal config
    if not hasattr(model, "config"):
        model.config = PretrainedConfig()
    model.config.model_type = "openflamingo"
    model.config.is_encoder_decoder = False

    # Add embedding accessors
    if not hasattr(model, "get_input_embeddings"):

        def _get(self):
            return self.lang_encoder.get_input_embeddings()

        def _set(self, new_emb):
            self.lang_encoder.set_input_embeddings(new_emb)

        model.get_input_embeddings = MethodType(_get, model)
        model.set_input_embeddings = MethodType(_set, model)

    return model


# ------------------------------------------------------------------
# ── MAIN ───────────────────────────────────────────────────────────
# ------------------------------------------------------------------
def generate_caption(
    model, tokenizer, image_processor, image_path, device=None, max_tokens=30
):
    """Generate a caption for validation"""
    try:
        # Get device from model if not provided
        if device is None:
            device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor(image)
        vision_x = (
            image_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
        )

        prompt = "<image><|endofchunk|>"
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = tokens.input_ids.shape[1]

        with torch.no_grad():
            # Initialize with prompt
            current_input_ids = tokens.input_ids
            current_attention_mask = tokens.attention_mask

            # Generate tokens one by one
            for _ in range(max_tokens):
                outputs = model(
                    vision_x=vision_x,
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                )

                # Get next token probabilities
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = next_token_logits.float()

                # Apply temperature and top-p sampling
                next_token_logits = next_token_logits / 0.8  # temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.9  # top-p threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to input
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_attention_mask = torch.ones_like(current_input_ids)

                # Stop if we generate EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Extract generated text
        generated_ids = current_input_ids[0][prompt_length:]
        generated_caption = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        del vision_x, tokens, current_input_ids, current_attention_mask
        torch.cuda.empty_cache()

        return generated_caption

    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Error generating caption"


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_model(model_size="medium"):
    """Load OpenFlamingo model, image processor, and tokenizer."""
    logger.info(f"Loading OpenFlamingo {model_size} model...")

    # Load model config and checkpoint first
    if model_size == "medium":
        model_path = "openflamingo/OpenFlamingo-4B-vitl-rpj3b"
        logger.info("Loaded ViT-L-14 model config.")
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    # Download and load checkpoint first
    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint_path = hf_hub_download(model_path, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Initialize model with proper CLIP weights
    from open_flamingo import create_model_and_transforms

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",  # Use OpenAI CLIP weights
        lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        cross_attn_every_n_layers=1,
    )

    # Load state dict with strict=False to handle any minor mismatches
    model.load_state_dict(checkpoint, strict=False)
    logger.info("Successfully loaded checkpoint")

    # Unfreeze the gating scalar to allow proper vision-language interaction
    if hasattr(model, "lang_to_vit_attention_gating"):
        model.lang_to_vit_attention_gating.requires_grad = True
        logger.info("Unfroze vision-language gating scalar")

    # Configure tokenizer
    logger.info("Configuring tokenizer with special tokens...")
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = 50279
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 0
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.bos_token_id = 0

    # Add custom tokens
    special_tokens = {
        "additional_special_tokens": ["<|endofchunk|>", "<image>"],
        "pad_token": "<PAD>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)

    # Log token configuration
    logger.info("Token configuration:")
    logger.info(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(
        f"  <|endofchunk|>: ID: {tokenizer.convert_tokens_to_ids('<|endofchunk|>')}"
    )

    return model, image_processor, tokenizer


def evaluate_model(model, val_dl, tokenizer, device, num_samples=100):
    """Comprehensive evaluation of the model"""
    model.eval()
    all_generated = []
    all_references = []
    total_loss = 0
    num_batches = 0

    # Get model's dtype
    dtype = next(model.parameters()).dtype
    logger.info(f"Using dtype: {dtype} for evaluation")

    # Get image processor from model
    image_processor = model.vision_encoder.image_processor
    logger.info("Using model's built-in image processor for evaluation")

    # Metrics
    bleu_scores = []
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dl:
            # Forward pass with correct dtype
            outputs = model(
                vision_x=batch["vision_x"].to(device, dtype=dtype),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )

            # Calculate loss
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate captions
            for i in range(min(len(batch["image_path"]), num_samples - total_samples)):
                try:
                    generated = generate_caption(
                        model,
                        tokenizer,
                        image_processor,
                        batch["image_path"][i],
                        device,
                    )
                    reference = batch["caption"][i]

                    # Store for BLEU calculation
                    all_generated.append(word_tokenize(generated.lower()))
                    all_references.append([word_tokenize(reference.lower())])

                    # Calculate exact matches
                    if generated.lower() == reference.lower():
                        exact_matches += 1
                    total_samples += 1

                except Exception as e:
                    logger.error(f"Error generating caption: {e}")
                    continue

                if total_samples >= num_samples:
                    break

            if total_samples >= num_samples:
                break

    # Calculate metrics
    avg_loss = total_loss / num_batches
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0

    # Calculate BLEU scores
    bleu_1 = corpus_bleu(all_references, all_generated, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(all_references, all_generated, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(all_references, all_generated, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(
        all_references, all_generated, weights=(0.25, 0.25, 0.25, 0.25)
    )

    return {
        "loss": avg_loss,
        "exact_match_rate": exact_match_rate,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "generated_samples": list(zip(all_generated, all_references)),
    }


def main() -> None:

    # ----------  runtime / hardware setup ----------
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Adjusted hyperparameters
    batch_size = 32
    epochs = 10
    lr = 1e-4  # Reduced from 2e-4 to 1e-4
    grad_acc = 2
    max_len = 100
    warmup_steps = 200  # Increased from 100 to 200
    max_grad_norm = 1.0

    out_dir = "./flamingo_lora_output"
    os.makedirs(out_dir, exist_ok=True)

    # ----------  distilled dataset paths ----------
    captions_json = "./vlm_distillation_output/distilled_dataset.json"
    images_dir = "./images"

    # ----------  load model and dataset ----------
    model, img_proc, tok = load_model("medium")
    model = patch_flamingo_for_peft(model)

    # ----------  attach LoRA ----------
    # First, let's inspect the model structure
    logger.info("Inspecting model structure for LoRA target modules...")
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Target attention layers in the language model
            if "lang_encoder.gpt_neox.layers" in name and (
                "query_key_value" in name  # Self-attention
                or "gated_cross_attn_layer.attn.to_" in name  # Cross-attention
            ):
                target_modules.append(name)
                logger.info(f"Selected for LoRA: {name}")

    logger.info(f"Selected {len(target_modules)} target modules for LoRA")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Increased rank for better adaptation
        lora_alpha=128,  # Increased alpha to match rank
        lora_dropout=0.1,  # Increased dropout for better regularization
        target_modules=target_modules,  # Use discovered module names
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device, dtype)

    # Warm-start the gating scalar
    if hasattr(model, "lang_to_vit_attention_gating"):
        model.lang_to_vit_attention_gating.data[:] = 1.0
        logger.info("Warm-started vision-language gating scalar to 1.0")

    # Log model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized:")
    logger.info(f"- Total parameters: {total:,}")
    logger.info(f"- Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    logger.info(f"- Device: {next(model.parameters()).device}")
    logger.info(f"- Dtype: {next(model.parameters()).dtype}")

    # ----------  datasets & dataloaders ----------
    full_ds = COCODataset(captions_json, images_dir, img_proc, tok, max_len=max_len)
    n = len(full_ds)
    train_ds, val_ds, _ = torch.utils.data.random_split(
        full_ds,
        [int(0.8 * n), int(0.1 * n), n - int(0.9 * n)],
        generator=torch.Generator().manual_seed(42),
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ----------  optimiser & scheduler ----------
    optim = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=0.01,  # Keep weight decay for regularization
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    opt_params = sum(p.numel() for p in optim.param_groups[0]["params"])
    print(f"Optimizer sees {opt_params:,} parameters")

    # Calculate total steps with gradient accumulation
    tot_steps = (len(train_dl) // grad_acc) * epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=tot_steps
    )

    # ----------  training loop ----------
    global_step = 0
    best_val_loss = float("inf")
    patience = 3
    no_improvement = 0
    validate_every = 1
    train_epoch_losses, val_epoch_losses = [], []
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")

        # Initialize gradients only once per gradient accumulation step
        optim.zero_grad()

        for step, batch in enumerate(pbar, 1):
            # Ensure vision input has correct shape (b, T_img, F, C, H, W)
            vis = batch["vision_x"].to(
                device, dtype
            )  # Already shaped correctly from dataset
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Log shapes for debugging
            if step == 1 and epoch == 1:
                logger.info(f"Input shapes:")
                logger.info(f"vision_x: {vis.shape}")  # Should be (b, 1, 1, C, H, W)
                logger.info(f"input_ids: {ids.shape}")
                logger.info(f"attention_mask: {att.shape}")
                logger.info(f"labels: {labels.shape}")

            # Forward pass
            out = model(
                vision_x=vis,
                input_ids=ids,
                attention_mask=att,
                labels=labels,
            )

            # Use model's loss computation which handles masking correctly
            loss = out.loss / grad_acc
            loss.backward()

            # Accumulate loss for logging
            running += loss.item() * grad_acc

            if step % grad_acc == 0 or step == len(train_dl):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()  # Only zero gradients after update
                global_step += 1

            # Update progress bar with more metrics
            pbar.set_postfix(
                {"loss": f"{running/step:.4f}", "lr": f"{sched.get_last_lr()[0]:.2e}"}
            )

            if step == 1 and epoch == 1:
                # Log initial loss without grad_acc division for consistent scale
                logger.info(f"Initial loss value: {out.loss.item():.4f}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        logger.debug(
                            f"Grad norm for {name}: {param.grad.norm().item()}"
                        )
                    else:
                        logger.debug(f"Grad for {name}: None")

        epoch_train_loss = running / len(train_dl)
        train_epoch_losses.append(epoch_train_loss)

        # Validation with comprehensive evaluation
        if (epoch + 1) % validate_every == 0:
            logger.info("Running validation...")
            metrics = evaluate_model(model, val_dl, tok, device)

            # Log metrics
            logger.info(f"Epoch {epoch} metrics:")
            logger.info(f"- Loss: {metrics['loss']:.4f}")
            logger.info(f"- Exact Match Rate: {metrics['exact_match_rate']:.4f}")
            logger.info(f"- BLEU-1: {metrics['bleu_1']:.4f}")
            logger.info(f"- BLEU-2: {metrics['bleu_2']:.4f}")
            logger.info(f"- BLEU-3: {metrics['bleu_3']:.4f}")
            logger.info(f"- BLEU-4: {metrics['bleu_4']:.4f}")

            # Log some sample generations
            logger.info("\nSample generations:")
            for gen, ref in metrics["generated_samples"][:3]:
                logger.info(f"Generated: {' '.join(gen)}")
                logger.info(f"Reference: {' '.join(ref[0])}")
                logger.info("-" * 50)

            # Early stopping check using BLEU-4 score
            if metrics["bleu_4"] > best_val_loss:
                best_val_loss = metrics["bleu_4"]
                best_metrics = metrics
                model.save_pretrained(f"{out_dir}/best_model")
                logger.info(
                    f"New best model saved with BLEU-4 = {metrics['bleu_4']:.4f}"
                )
                no_improvement = 0
            else:
                no_improvement += 1
                logger.info(f"No improvement for {no_improvement} validation checks")

            if no_improvement >= patience:
                logger.info(
                    f"Early stopping after {patience} epochs without improvement"
                )
                break

            val_epoch_losses.append(metrics["loss"])

        # Save checkpoint each epoch
        model.save_pretrained(f"{out_dir}/checkpoint_epoch_{epoch}")

        # Monitor GPU usage after clearing cache
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

        gc.collect()

    # Save final model and best metrics
    model.save_pretrained(f"{out_dir}/final_model")
    if best_metrics:
        with open(f"{out_dir}/best_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
    logger.info("✓ training finished and model saved")

    # Plot training curves
    min_len = min(len(train_epoch_losses), len(val_epoch_losses))
    epochs_range = range(1, min_len + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_epoch_losses[:min_len], label="Train Loss")
    plt.plot(epochs_range, val_epoch_losses[:min_len], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("OpenFlamingo LoRA Fine-tune — Loss per Epoch")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png")
    plt.close()


# run it
if __name__ == "__main__":
    logger.info(f"Starting run at {datetime.now()}")
    main()
