# main.py

import argparse
from torch import set_warn_always
set_warn_always(False) 

from config import Config
from train_manager import Trainer
from my_nn import cuda_optim

def parse_optional_float(value):
    """Helper to handle arguments that can be a float or None"""
    if str(value).lower() == 'none':
        return None
    return float(value)

def parse_optional_int(value):
    """Helper to handle arguments that can be an int or None"""
    if str(value).lower() == 'none':
        return None
    return int(value)

def main():
    # 1. Setup Device
    device = cuda_optim.setup_cuda_optimizations()
    
    # 2. Initialize Default Config
    cfg = Config()

    # 3. Define Standard Parser
    parser = argparse.ArgumentParser(description="Training Configuration")

    # --- 1. Paths & Infrastructure ---
    parser.add_argument('--data_root', type=str, default=cfg.data_root, help='Path to data')
    parser.add_argument('--save_dir', type=str, default=cfg.save_dir, help='Path to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=cfg.log_dir, help='Path to save logs')
    parser.add_argument('--seed', type=int, default=cfg.seed, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers, help='Dataloader workers')

    # --- 2. Training Loop ---
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Batch size')
    parser.add_argument('--save_every', type=int, default=cfg.save_every, help='Epoch interval to save')
    parser.add_argument('--visualize_every', type=int, default=cfg.visualize_every, help='Epoch interval to visualize')
    parser.add_argument('--cosine_epoch', type=int, default=cfg.cosine_epoch, help='Epochs for cosine scheduler')
    parser.add_argument('--accuracy_threshold', type=float, default=cfg.accuracy_threshold, help='Target accuracy')

    # --- 3. Optimization ---
    parser.add_argument('--learning_rate', type=float, default=cfg.learning_rate, help='Base Encoder LR')
    parser.add_argument('--lr_ratio', type=float, default=cfg.lr_ratio, help='Ratio for metric layer LR')
    parser.add_argument('--weight_decay', type=float, default=cfg.weight_decay, help='Optimizer weight decay')
    parser.add_argument('--momentum', type=float, default=cfg.momentum, help='Optimizer momentum')
    
    # Decoder specific
    parser.add_argument('--decoder_learning_rate', type=float, default=cfg.decoder_learning_rate)
    parser.add_argument('--decoder_weight_decay', type=float, default=cfg.decoder_weight_decay)

    # --- Fine Tuning ---
    parser.add_argument('--fine_tunning_lr', type=float, default=cfg.fine_tunning_lr)
    parser.add_argument('--fine_tunning_epochs', type=int, default=cfg.fine_tunning_epochs)

    # --- 4. Model Architecture ---
    parser.add_argument('--embedding_dim', type=int, default=cfg.embedding_dim)
    parser.add_argument('--dropout', type=float, default=cfg.dropout)
    parser.add_argument('--total_classes', type=int, default=cfg.total_classes)

    # --- 5. Loss Hyperparameters ---
    parser.add_argument('--loss_pixel', type=float, default=cfg.loss_pixel, help='Reconstruction loss weight')
    parser.add_argument('--scale_factor', type=float, default=cfg.scale_factor, help='ArcFace scale (s)')
    parser.add_argument('--angular_margin', type=float, default=cfg.angular_margin, help='ArcFace margin (m)')
    parser.add_argument('--warmup_epochs', type=int, default=cfg.warmup_epochs)

    # --- 6. Dataset ---
    parser.add_argument('--image_size', type=int, default=cfg.image_size)
    

    parser.add_argument('--train_sample', type=int, default=cfg.train_sample)
    parser.add_argument('--val_sample', type=int, default=cfg.val_sample)

    # Handling Lists
    # usage: --dataset_name_train MNIST FashionMNIST
    parser.add_argument('--dataset_name_train', nargs='+', default=cfg.dataset_name_train)
    parser.add_argument('--dataset_name_test', nargs='+', default=cfg.dataset_name_test)


    # 4. Parse and Update
    args = parser.parse_args()

    # Update the config object with parsed arguments
    # We iterate over the args to automatically update the cfg object
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # Optional: Print to verify
    print("Training with Config:")
    print(f"LR: {cfg.learning_rate}, Batch: {cfg.batch_size}, Data: {cfg.dataset_name_train}")

    # 5. Run Trainer
    train = Trainer(cfg=cfg, device=device)
    train.run()

if __name__ == "__main__":
    main()