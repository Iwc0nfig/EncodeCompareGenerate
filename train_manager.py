
import os
import torch
from pathlib import Path
from typing import Optional, Tuple

from config import Config
from my_nn import MyEncoder, MyDecoder, ArcFace
from train.train_encoder import EncoderTraining
from train.train_decoder import DecoderTraining
from train.finetunning import FineTuning
from My_Dataloader import create_siamese_dataloaders , _load_datasets



class Trainer:
    """
    High-level training manager that builds data, models, handles checkpoints,
    and runs the encoder/decoder training flows.

    Parameters
    ----------
    cfg : Config
        Configuration object (instance of your project's Config dataclass).
    device : str | torch.device, optional
        Device string like "cuda" or "cpu". Defaults to automatic detection.
    resume : bool, optional
        If True, attempt to resume from checkpoints when available.
    """

    def __init__(self, cfg: Config, device: Optional[str] = None, resume: bool = True):
        self.cfg = cfg
        self.device = device
        self.resume = resume

        # placeholders filled during setup()
        self.train_loader = None
        self.val_loader = None
        self.total_classes = None

        self.encoder :torch.nn.Module = None
        self.metric_fc :torch.nn.Module= None
        self.decoder :torch.nn.Module = None

        # ensure save/log dirs exist
        Path(self.cfg.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Setup helpers
    # -------------------------
    def setup_data(self) -> None:
        _load_datasets(self.cfg)
        train_loader, val_loader, total_classes = create_siamese_dataloaders(self.cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_classes = total_classes
        self.cfg.total_classes = self.total_classes
        print(f"[=======] Tottal Classes are :{self.total_classes}")

    def setup_models(self) -> None:
        """
        Instantiate encoder, metric (ArcFace), and decoder and move them to device.
        """
        print("[Trainer] Building models...")
        # adapt constructor names/args to your my_nn definitions
        self.encoder = MyEncoder(
            emb_dim=self.cfg.embedding_dim, 
            in_channels=self.cfg.num_input_channels
            )
        
        self.metric_fc = ArcFace(
            in_features=self.cfg.embedding_dim, 
            out_features=self.total_classes, 
            s=self.cfg.scale_factor, 
            m=self.cfg.angular_margin
        )

        self.decoder = MyDecoder(
            emb_dim=self.cfg.embedding_dim, 
            out_channels=self.cfg.num_input_channels
        )

        # move to device
        self.encoder.to(self.device)
        self.metric_fc.to(self.device)
        self.decoder.to(self.device)

    # -------------------------
    # Checkpoint helpers
    # -------------------------
    def _path(self, name: str) -> str:
        """Helper to build full path from cfg.save_dir and filename attribute names."""
        return os.path.join(self.cfg.save_dir, getattr(self.cfg, name))

    

    def save_checkpoints(self) -> None:
        """Save encoder, metric_fc, and decoder to cfg.save_dir using cfg paths."""
        print("[Trainer] Saving final checkpoints...")
        torch.save(self.encoder.state_dict(), self._path("encoder_path"))
        torch.save(self.metric_fc.state_dict(),self._path("matric_fc_path"))
        torch.save(self.decoder.state_dict(), self._path("decoder_path"))
        print("[Trainer] Checkpoints saved.")

    # -------------------------
    # High level train actions
    # -------------------------
    def train_encoder(self) -> None:
        """
        Run encoder training using EncoderTraining class.
        Expects EncoderTraining signature: EncoderTraining(encoder, metric_fc, device, loader, config)
        If your project's EncoderTraining expects different args, you may adapt this method.
        """
        print("[Trainer] Starting encoder training phase...")
        trainer = EncoderTraining(
            encoder=self.encoder,
            metric_fc=self.metric_fc,
            device=self.device,
            train_loader=self.train_loader,
            val_loader= self.val_loader,
            config=self.cfg
        )
        
        trainer.run()
        print("[Trainer] Encoder training finished.")

    def train_decoder(self) -> None:
        """
        Run decoder training using DecoderTraining class.
        Expects DecoderTraining signature: DecoderTraining(encoder, decoder, device, loader, config)
        """
        print("[Trainer] Starting decoder training phase...")
        trainer = DecoderTraining(
            encoder=self.encoder,
            decoder=self.decoder,
            device=self.device,
            train_loader=self.train_loader,
            val_loader= self.val_loader,
            config=self.cfg
        )
        trainer.run()
        print("[Trainer] Decoder training finished.")


    def train_finetune(self) -> None:
        print("[Trainer] Starting fine-tuning phase...")
        
        # --- CRITICAL FIX: RELOAD WEIGHTS ---
        # This ensures we strip any torch.compile artifacts and improper frozen states
        # from the previous phases.
        print("[Trainer] Reloading best Encoder/Decoder weights for Fine-Tuning...")

        def clean_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove the prefix added by torch.compile
                if k.startswith("_orig_mod."):
                    new_state_dict[k.replace("_orig_mod.", "")] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict
            
        # Helper to get the actual model if it is wrapped by torch.compile
        def get_model_ref(model):
            if hasattr(model, "_orig_mod"):
                return model._orig_mod
            return model
        

        # Load Encoder (Saved from Phase 1)
        enc_path = self._path("encoder_path")
        if os.path.exists(enc_path):
            print(f"Loading Encoder from: {enc_path}")
            checkpoint = torch.load(enc_path, map_location=self.device , weights_only=True)
            # FIX: Load clean weights into the UNWRAPPED model
            get_model_ref(self.encoder).load_state_dict(clean_state_dict(checkpoint))
            
        dec_path = self._path("decoder_path")
        if os.path.exists(dec_path):
            print(f"Loading Decoder from: {dec_path}")
            checkpoint = torch.load(dec_path, map_location=self.device, weights_only=True)
            # FIX: Load clean weights into the UNWRAPPED model
            get_model_ref(self.decoder).load_state_dict(clean_state_dict(checkpoint))

        met_path = self._path("matric_fc_path") 
        if os.path.exists(met_path):
            print(f"Loading Metric FC from: {met_path}")
            checkpoint = torch.load(met_path, map_location=self.device, weights_only=True)
            get_model_ref(self.metric_fc).load_state_dict(clean_state_dict(checkpoint))
        else:
            print(f"WARNING: Metric FC checkpoint not found at {met_path}")
        

        trainer = FineTuning(
            encoder=self.encoder,
            decoder=self.decoder,
            metric_fc=self.metric_fc,
            device=self.device,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.cfg
        )
        trainer.run()
        print("[Trainer] Fine-tuning finished.")
    # -------------------------
    # Public runner
    # -------------------------
    def run(self) -> None:
        """
        Entry point for running training. 
        Catches KeyboardInterrupt to allow skipping phases manually.
        """

        # 1) Data
        self.setup_data()

        # 2) Models
        self.setup_models()



        # 3) Run phases
        
        # --- PHASE 1: ENCODER ---
        try:
            self.encoder = torch.compile(self.encoder, mode="default")
            self.metric_fc = torch.compile(self.metric_fc, mode="default")
            self.train_encoder()
        except KeyboardInterrupt:
            print("\n[Trainer] >>> Interrupted! Saving current state and moving to Decoder... <<<")
        finally:
            # ALWAYS save, even if interrupted
            print("[Trainer] Saving Encoder & Metric FC...")
            torch.save(self.encoder.state_dict(), self._path("encoder_path"))
            torch.save(self.metric_fc.state_dict(), self._path("matric_fc_path"))

        # --- PHASE 2: DECODER ---
        try:
            self.decoder = torch.compile(self.decoder, mode="default")
            self.train_decoder()
        except KeyboardInterrupt:
            print("\n[Trainer] >>> Interrupted! Saving current state and moving to Fine-tuning... <<<")
        finally:
            # ALWAYS save, even if interrupted
            print("[Trainer] Saving Decoder...")
            torch.save(self.decoder.state_dict(), self._path("decoder_path"))

        # --- PHASE 3: FINE-TUNING ---
        try:
            self.train_finetune()
        except KeyboardInterrupt:
            print("\n[Trainer] >>> Interrupted! Skipping remaining Fine-tuning... <<<")

        # 5) Save final checkpoints
        try:
            self.save_checkpoints()
        except KeyboardInterrupt:
            print("\n[Trainer] Saving interrupted, but attempting to finish...")
            
        print("[Trainer] Run complete.")
