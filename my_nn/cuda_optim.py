
import numpy as np
import torch
import torch.nn as nn
"""
This code is optimzed to my setup I have an ada rtx 4000 sff 20gb 
Feel free to modify the code for your gpu setup
If you have and Nvidia gpu after 2020 you don't need to change anything 
"""

def setup_cuda_optimizations():
    if torch.cuda.is_available():
        # Prefer faster matmul kernels (Ampere/Ada benefit)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Enable optimizations for RTX Ada series
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy for 20GB VRAM
        torch.cuda.empty_cache()
        
        
        return "cuda"
    else:
        print("CUDA not available, using CPU")
        return "cpu"

