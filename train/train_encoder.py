
import torch
import torch.nn as nn
import os
from my_nn import  MyEncoder, ArcFace
from config import Config
from My_Dataloader import my_Dataloader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR



class EncoderTraining():
    def __init__(
            self,
            encoder:MyEncoder,
            metric_fc:ArcFace,
            device:torch.device,
            train_loader:my_Dataloader,
            val_loader:my_Dataloader,
            config:Config
        ):

        
        self.encoder = encoder
        self.encoder.train()
        self.metric_fc = metric_fc
        self.metric_fc.train()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.cfg = config
        
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.criterion_cls = nn.CrossEntropyLoss()


        self.current_epoch = 0
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)

        
    def _build_optimizer(self):
        def get_parameter_groups(module, lr, weight_decay):
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
            self.encoder, 
            lr=self.cfg.learning_rate, 
            weight_decay=self.cfg.weight_decay
        )

        
        metric_groups = get_parameter_groups(
            self.metric_fc, 
            lr=self.cfg.learning_rate * self.cfg.lr_ratio, 
            weight_decay=self.cfg.weight_decay 
        )

        # 3. Combine and Build
        optimizer = torch.optim.SGD(
            encoder_groups + metric_groups,
            momentum=self.cfg.momentum,
        )
        
        return optimizer

    def _build_scheduler(self):
        # 1. Define the Warmup Scheduler
        # LinearLR scales the LR by a factor. 
        # We calculate start_factor so that: initial_lr = peak_lr * start_factor
        

        start_factor = self.cfg.warmup_lr / self.cfg.learning_rate
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
            optimizer=self.optimizer,
            T_max=cosine_steps,   # total number of decay steps
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

    def _update_arcface_margin(self):
        """
        Handles the Linear Warmup of the ArcFace Margin.
        Returns the current margin for logging purposes.
        """
        current_margin = self.cfg.angular_margin

        if self.current_epoch < self.cfg.warmup_epochs:
            ratio = self.current_epoch / self.cfg.warmup_epochs
            current_margin = self.cfg.angular_margin * ratio
            self.metric_fc.set_margin(current_margin)

        elif self.current_epoch == self.cfg.warmup_epochs:
            self.metric_fc.set_margin(self.cfg.angular_margin)
            
        return current_margin



    def validate(self):
        self.encoder.eval()
        self.metric_fc.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img, label = batch[0].to(self.device), batch[1].to(self.device)
                embeddings = self.encoder(img)
                # For ArcFace validation, we often use Cosine Sim, 
                # but here we check classification accuracy on valid set as proxy
                logits = self.metric_fc(embeddings, label=None) 
                _, predicted = torch.max(logits.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        return 100. * correct / total
        

    def fit(self):
        total_loss = 0
        total_samples= 0
        correct = 0

        for batch in self.train_loader:
            img, label = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            # Forward (Encoder Only)
            embeddings = self.encoder(img)
            logits = self.metric_fc(embeddings, label)

            # Loss
            loss = self.criterion_cls(logits, label)

            # Backward
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.metric_fc.parameters(), 5.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += label.size(0)
            correct += (predicted == label).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total_samples

        return avg_loss, acc
            


    def run(self):
        """
        Dynamic Training Loop:
        Stops when Config.accuracy_threshold is reached on validation set.
        """
        print(f"--- Starting Encoder Training (Goal: >{self.cfg.accuracy_threshold}% Acc) ---")
        

        while True:
            margin = self.cfg.angular_margin # self._update_arcface_margin()
            
            train_loss, train_acc = self.fit()
            val_acc = self.validate()

            total_scheduled_epochs = self.cfg.warmup_epochs + self.cfg.cosine_epoch

            # Logic: Step normally if inside the window, otherwise clamp manually
            if self.current_epoch < total_scheduled_epochs:
                self.scheduler.step()
            else:
                # --- FIX: Update all 4 groups manually ---
                # Groups 0 & 1 are Encoder
                self.optimizer.param_groups[0]['lr'] = self.cfg.min_lr
                self.optimizer.param_groups[1]['lr'] = self.cfg.min_lr
                
                # Groups 2 & 3 are Metric FC
                self.optimizer.param_groups[2]['lr'] = self.cfg.min_lr * self.cfg.lr_ratio
                self.optimizer.param_groups[3]['lr'] = self.cfg.min_lr * self.cfg.lr_ratio

            # --- FIX: Log Index 0 (Encoder) and Index 2 (Metric) ---
            current_encoder_lr = self.optimizer.param_groups[0]['lr']
            current_metric_lr = self.optimizer.param_groups[2]['lr'] 
            self.writer.add_scalar("LR/Encoder", current_encoder_lr, self.current_epoch)
            self.writer.add_scalar("LR/MetricFC", current_metric_lr, self.current_epoch)
            self.writer.add_scalar("Train/Loss",train_loss, self.current_epoch)
            self.writer.add_scalar("Train/Acc",train_acc,self.current_epoch)
            self.writer.add_scalar("Validation/Acc", val_acc ,self.current_epoch)

            print(f"Epoch {self.current_epoch:03d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Margin: {margin:.2f}")


            # Dynamic Threshold Check
            if val_acc >= self.cfg.accuracy_threshold or self.current_epoch> total_scheduled_epochs + self.cfg.final_epoch:
                print(f"Target Accuracy Reached ({val_acc:.2f}%)! Stopping Encoder training.")
                break
            
            self.current_epoch += 1

        




