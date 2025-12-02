import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from config import Config
from my_nn import RichNet
from My_Dataloader import create_siamese_dataloaders

def denormalize(tensor, mean, std):
    """
    Reverses the normalization applied by the Dataloader 
    so the image looks natural when plotted.
    """
    # Clone to avoid modifying the original tensor
    img = tensor.clone() 
    
    # Reverse (img - mean) / std  -->  img * std + mean
    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
        
    img.mul_(std).add_(mean)
    return img.clamp(0, 1) # Ensure valid pixel range

def load_models(cfg:Config, device):
    """
    Loads the RichNet and populates weights from checkpoints.
    Handles '_orig_mod.' prefix from torch.compile().
    """
    print("--- Loading Models ---")
    model = RichNet(in_channels=cfg.num_input_channels, emb_dim=cfg.embedding_dim).to(device)
    
    # Helper to clean state_dict keys
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove the prefix added by torch.compile
            if k.startswith("_orig_mod."):
                new_state_dict[k.replace("_orig_mod.", "")] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # 1. Load Encoder
    enc_path = os.path.join(cfg.save_dir, cfg.encoder_path)
    if os.path.exists(enc_path):
        print(f"Loading Encoder from: {enc_path}")
        checkpoint = torch.load(enc_path, map_location=device)
        model.encoder.load_state_dict(clean_state_dict(checkpoint))
    else:
        print(f"[WARNING] Encoder checkpoint not found at {enc_path}")

    # 2. Load Decoder
    config_dec_path = os.path.join(cfg.save_dir, cfg.decoder_path)
    
   
        
    if config_dec_path:
        print(f"Loading Decoder from: {config_dec_path}")
        checkpoint = torch.load(config_dec_path, map_location=device)
        model.decoder.load_state_dict(clean_state_dict(checkpoint))
    else:
        print(f"[WARNING] Decoder checkpoint not found!")

    model.eval()
    return model

def run_test(num_samples=3):
    # 1. Setup
    cfg = Config()
    device = "cpu"
    print(f"Running Inference on: {device}")

    # 2. Get Data (Test Set)
    # create_siamese_dataloaders returns (train_loader, val_loader, offset)
    # We only need the validation loader.
    _, val_loader = create_siamese_dataloaders(cfg, test=True) # Note: returns (None, loader)
    
    # 3. Load Model
    model = load_models(cfg, device)

    # 4. Get a batch of images
    print("Fetching random batch...")
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    
    # Select N random indices from this batch
    indices = random.sample(range(len(images)), num_samples)

    # 5. Visualization Loop
    plt.figure(figsize=(10, 4 * num_samples))
    
    # Get Norm stats for visualization
    mean, std = cfg.get_norm_stats("MNIST") 

    for i, idx in enumerate(indices):
        input_img = images[idx].unsqueeze(0).to(device) # Add batch dim [1, 1, 28, 28]
        label_id = labels[idx].item()

        with torch.no_grad():
            # A. ENCODER: Draw me a concept (z)
            z_orig = model.encoder(input_img)
            
            # B. DECODER: Draw the image based on the concept
            recon_img = model.decoder(z_orig)
            
            # C. VERIFY: Pass reconstruction back to encoder
            z_recon = model.encoder(recon_img)

            # Calculate Similarity (Is this a 9?)
            # 1.0 = Perfect Match, 0.0 = Unrelated, -1.0 = Opposite
            sim_score = F.cosine_similarity(z_orig, z_recon).item() * 100

        # --- Plotting ---
        # Un-normalize for pretty display
        img_vis_orig = denormalize(input_img[0].cpu(), mean, std).permute(1, 2, 0).numpy()
        img_vis_recon = recon_img[0].cpu().permute(1, 2, 0).numpy() # Decoder output is usually Sigmoid (0-1)

        # Original
        ax = plt.subplot(num_samples, 2, 2*i + 1)
        ax.imshow(img_vis_orig, cmap='gray')
        ax.set_title(f"Input (Label: {label_id})")
        ax.axis('off')

        # Reconstruction
        ax = plt.subplot(num_samples, 2, 2*i + 2)
        ax.imshow(img_vis_recon, cmap='gray')
        
        # Color code the title based on success
        color = 'green' if sim_score > 90 else 'red'
        status = "MATCH" if sim_score > 90 else "MISMATCH"
        
        ax.set_title(f"Reconstruction\nParams match: {sim_score:.2f}% ({status})", color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print("\nTest Complete. If the images look similar and score is >90%, the Cycle is consistent.")

if __name__ == "__main__":
    run_test(num_samples=3)