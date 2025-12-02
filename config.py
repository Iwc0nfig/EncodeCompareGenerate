from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class Config:
    # -------------------------------------------------------------------------
    # 1. Paths & Infrastructure
    # -------------------------------------------------------------------------
    data_root: str = "./data"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    encoder_path:str = "encoder.pth"
    decoder_path:str = "decoder.pth"
    matric_fc_path:str = "metric.pth"

    seed: int = 42
    num_workers: int = 10 #consider modify these to you cpu cores 


    # -------------------------------------------------------------------------
    # 2. Training Loop
    # -------------------------------------------------------------------------
    batch_size: int = 512 #This will required around 4Gb of your vram
    save_every: int = 10
    visualize_every: int = 5
    cosine_epoch: int = 100
    final_epoch : int = 15
    accuracy_threshold: float = 98.0
    

    # -------------------------------------------------------------------------
    # 3. Optimization & Scheduler
    # -------------------------------------------------------------------------
    warmup_lr:float = 5e-3
    min_lr: float = 1e-5
    learning_rate: float = 0.1 # Base LR (Encoder)
    lr_ratio : float = 1.0  # Metric FC will be Base * lr_ratio
    
    weight_decay: float = 5e-4
    momentum:float = 0.9

    decoder_warmup_lr: float = 1e-4
    decoder_learning_rate: float = 1e-3
    decoder_weight_decay:float = 1e-5    

    # -------------------------------------------------------------------------
    # Fine Tunning 
    # -------------------------------------------------------------------------

    fine_tunning_lr:float = 5e-6
    fine_tunning_epochs:int = 10
    
    
    

    # -------------------------------------------------------------------------
    # 4. Model Architecture
    # -------------------------------------------------------------------------
    embedding_dim: int =  128
    dropout: float = 0.2
    num_input_channels: int = 1
    total_classes: int = 10
    
    # -------------------------------------------------------------------------
    # 5. Loss Hyperparameters
    # -------------------------------------------------------------------------
    # Reconstruction Loss Weight

    loss_pixel: float = 0.6   # How much the model should worry about pixel perfection
    
    # Metric Learning for ArcFace
    scale_factor: float = 32.0 
    angular_margin: float = 0.50 
    warmup_epochs : int = 15

    # -------------------------------------------------------------------------
    # 6. Dataset Configuration
    # -------------------------------------------------------------------------
    image_size: int = 28
    emnist_split: str = "letters"

    train_sample : int = 6000
    val_sample: int = 100

    dataset_name_train: List[str] = field(default_factory=lambda: [
        "MNIST",
        #"EMNIST"
    ]) 

    dataset_name_test: List[str] = field(default_factory=lambda: [
        "MNIST",
        #"EMNIST"
    ]) 

    # -------------------------------------------------------------------------
    # 7. Helper Methods
    # -------------------------------------------------------------------------
    def get_norm_stats(self, name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Returns a universal normalization stat for all datasets.
        Scales pixel values from [0, 1] to [-1, 1].
        """
        return ((0.1307 ), (0.3081,))
    


