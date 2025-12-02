
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from my_nn import  MyDecoder,MyEncoder
from My_Dataloader import my_Dataloader
from config import Config




class DecoderTraining:
    def __init__(
            self,
            encoder:MyEncoder,
            decoder:MyDecoder,
            device:torch.device,
            train_loader:my_Dataloader,
            val_loader:my_Dataloader,
            config: Config
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config


        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.criterion_recon = nn.MSELoss()


        self.current_epoch = 0
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        



    def _build_optimizer(self):
        def get_parameter_groups(module:nn.Module, lr, weight_decay):
            decay_params = []
            no_decay_params = []
            
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                
                # TRICK: Generally, parameters with 1 dimension (biases, BN weights)
                # should NOT have weight decay.
                # Parameters with >1 dimension (Conv2d weights, Linear weights) SHOULD.
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
            return [
                {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
            ]

        # 1. Get Encoder Groups (Base LR)
        encoder_groups = get_parameter_groups(
            self.decoder, 
            lr=self.cfg.decoder_learning_rate, 
            weight_decay=self.cfg.decoder_weight_decay
        )

        
        

        # 3. Combine and Build
        optimizer = torch.optim.AdamW(
            encoder_groups 
        )
        
        return optimizer
    

    def _build_scheduler(self):
        # 1. Define the Warmup Scheduler
        # LinearLR scales the LR by a factor. 
        # We calculate start_factor so that: initial_lr = peak_lr * start_factor
        start_factor = self.cfg.decoder_warmup_lr / self.cfg.decoder_learning_rate
        
        scheduler_warmup = LinearLR(
            self.optimizer, 
            start_factor=start_factor, 
            end_factor=1.0, 
            total_iters=self.cfg.warmup_epochs
        )

        # 2. Define the Cosine Scheduler
        # Important: T_max should be the remaining epochs AFTER warmup, 
        # not the total epochs.
        cosine_steps = self.cfg.cosine_epoch 
        
        scheduler_cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=cosine_steps, 
            eta_min=self.cfg.min_lr
        )

        # 3. Combine them
        # The scheduler automatically switches from warmup to cosine 
        # once 'milestones' is reached.
        scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[scheduler_warmup, scheduler_cosine], 
            milestones=[self.cfg.warmup_epochs]
        )
        
        return scheduler


    def validate(self):
        """
        Validate based on Goal.txt: 
        "Similarity of the embedding to the embeddings of the reconstructed image"
        """
        self.encoder.eval()
        self.decoder.eval()
        
        total_similarity = 0.0
        batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img = batch[0].to(self.device)
                
                # 1. Get original embedding
                z_orig = self.encoder(img)
                
                # 2. Reconstruct image
                img_recon = self.decoder(z_orig)
                
                # 3. Get embedding of reconstructed image
                z_recon = self.encoder(img_recon)
                
                # 4. Calculate Cosine Similarity
                # Normalize both to check directional similarity
                z_orig_norm = F.normalize(z_orig, p=2, dim=1)
                z_recon_norm = F.normalize(z_recon, p=2, dim=1)
                
                sim = (z_orig_norm * z_recon_norm).sum(dim=1).mean()
                total_similarity += sim.item()
                batches += 1
                
        return (total_similarity / batches) * 100

    def fit(self):
        self.encoder.eval()
   
        self.decoder.train()

        # 2. Manage Gradients
        for param in self.encoder.parameters(): 
            param.requires_grad = False

        for param in self.decoder.parameters(): 
            param.requires_grad = True


        
        total_loss = 0.0

        for batch in self.train_loader:
            img, _ = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            # Forward Encoder (No Grad - Treat as constant)
            with torch.no_grad():
                embeddings = self.encoder(img)

            # Forward Decoder
            recon = self.decoder(embeddings)

            # Loss
            loss = self.criterion_recon(recon, img)

            # Backward
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss



    def run(self):
        print("--- Starting Decoder Training (Reconstruction Phase) ---")
        
        while True:

            loss = self.fit()
            sim_score = self.validate()

            total_scheduled_epochs = self.cfg.warmup_epochs + self.cfg.cosine_epoch
     
            if self.current_epoch < total_scheduled_epochs:
                self.scheduler.step()
            else:
                self.optimizer.param_groups[0]['lr'] = self.cfg.min_lr
                
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("LR_per_epoch", current_lr, self.current_epoch)
            self.writer.add_scalar('Loss/Reconstruction', loss, self.current_epoch)
            self.writer.add_scalar("Sim Score", sim_score , self.current_epoch)

            print(f"Epoch {self.current_epoch:03d} | Recon Loss: {loss:.4f} | Emb Similarity: {sim_score:.2f}")


            if loss > self.cfg.loss_pixel or self.current_epoch> total_scheduled_epochs + self.cfg.final_epoch : 
                print("High fidelity reconstruction achieved.")
                break

            self.current_epoch +=1