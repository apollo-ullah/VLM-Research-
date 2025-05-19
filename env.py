# ============================================================================
# Clean LLaVA v1.1.1 Environment Setup for Jupyter
# ============================================================================

# Print system information
import os
import sys
import torch
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Fix CUDA library path (important for bitsandbytes)
cuda_lib_path = "/usr/local/cuda/lib64"
if os.path.exists(cuda_lib_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"Added {cuda_lib_path} to LD_LIBRARY_PATH")

# Create workspace directory if needed
workspace_dir = "/workspace"
!mkdir -p {workspace_dir}
%cd {workspace_dir}
print(f"Working in {os.getcwd()}")

# ============================================================================
# STEP 1: THOROUGH CLEANUP - Remove ALL existing LLaVA packages
# ============================================================================
print("\n========== 1. THOROUGH CLEANUP ==========")
print("Removing all existing LLaVA and conflicting packages...")
!pip uninstall -y llava llava-torch llava-hf llava-1.5 llava-vicuna llava-13b llava-7b llava* 
!pip uninstall -y open_flamingo open-flamingo transformers accelerate peft bitsandbytes flash-attn
!pip uninstall -y pydantic evaluate spacy nltk
!pip cache purge

# ============================================================================
# STEP 2: Install core PyTorch and transformers dependencies
# ============================================================================
print("\n========== 2. INSTALLING CORE DEPENDENCIES ==========")
!pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.31.0 "pydantic<2" sentencepiece protobuf
!pip install bitsandbytes==0.40.2 numpy==1.24.3

# ============================================================================
# STEP 3: Install CLIP and other utilities needed by distill.py
# ============================================================================
print("\n========== 3. INSTALLING CLIP AND UTILITIES ==========")
# Note the --no-deps flag to prevent pulling in unwanted dependencies
!pip install --no-deps git+https://github.com/openai/CLIP.git
!pip install tqdm matplotlib huggingface_hub

# ============================================================================
# STEP 4: Set up LLaVA v1.1.1 from source
# ============================================================================
print("\n========== 4. SETTING UP LLAVA v1.1.1 ==========")
# Install LLaVA-specific dependencies
!pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13

# Clone and install LLaVA v1.1.1
LLAVA_DIR = "/workspace/LLaVA"
print(f"Cloning LLaVA v1.1.1 to {LLAVA_DIR}...")
!rm -rf {LLAVA_DIR}
!git clone https://github.com/haotian-liu/LLaVA.git {LLAVA_DIR}
%cd {LLAVA_DIR}
!git checkout v1.1.1
!pip install -e . --no-build-isolation
%cd {workspace_dir}

# ============================================================================
# STEP 5: Verify correct LLaVA installation
# ============================================================================
print("\n========== 5. VERIFYING LLAVA INSTALLATION ==========")

# Add LLaVA to Python path
if LLAVA_DIR not in sys.path:
    sys.path.insert(0, LLAVA_DIR)
    print(f"Added {LLAVA_DIR} to Python path")

import inspect
try:
    import llava
    from llava.model.builder import load_pretrained_model
    
    # Print path to verify we're using the right version
    import pathlib
    llava_path = pathlib.Path(llava.__file__).parent
    print(f"LLaVA path: {llava_path}")
    
    # Check builder signature to confirm correct version
    sig = inspect.signature(load_pretrained_model)
    params = list(sig.parameters.keys())
    print(f"Builder signature: {sig}")
    
    if params[0] == 'model_path':
        print("\n✅ SUCCESS: LLaVA v1.1.1 correctly installed!")
    else:
        print("\n❌ ERROR: Wrong LLaVA version detected!")
        print(f"Expected first parameter: 'model_path', found: '{params[0]}'")
except ImportError as e:
    print(f"LLaVA import error: {e}")
    print("LLaVA not properly installed - check installation steps")

# Brief test of CLIP which is needed for distill.py
try:
    print("\nTesting CLIP...")
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    print("CLIP processor loaded successfully!")
except Exception as e:
    print(f"Error loading CLIP: {e}")

print("\n========== SETUP COMPLETE ==========")
print("You can now run distill.py to use LLaVA v1.1.1 for VLM assessment!")
print("Example: python distill.py 10  # Process 10% of the dataset")