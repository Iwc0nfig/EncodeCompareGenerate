import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from my_nn import MyEncoder, MyDecoder, ArcFace
from My_Dataloader import my_Dataloader
from config import Config

class FineTuning:
    def __init__(
            self,
            encoder: MyEncoder,
            decoder: MyDecoder,
            metric_fc: ArcFace,
            device: torch.device,
            train_loader: my_Dataloader,
            val_loader: my_Dataloader,
            config: Config
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.metric_fc = metric_fc
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config

        # 1. Unfreeze first
        self._unfreeze_models()
        
        # 2. Then build optimizer (so it sees the requires_grad=True parameters)
        self.optimizer = self._build_optimizer()

        self.criterion_recon = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()

        self.current_epoch = 0
        self.writer = SummaryWriter(log_dir=f"{self.cfg.log_dir}/finetune")

    def _unfreeze_models(self):
        # We unfreeze encoder and decoder
        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True

        # Keep Metric FC frozen (Centers fixed) - Standard for fine-tuning
        for param in self.metric_fc.parameters():
            param.requires_grad = False

    def _build_optimizer(self):
        def get_parameter_groups(module:nn.Module, lr, weight_decay):
            decay_params = []
            no_decay_params = []    
            for name , param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            return [
                {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
            ]

        encoder_groups = get_parameter_groups(
            self.encoder, 
            lr=self.cfg.fine_tunning_lr, 
            weight_decay=self.cfg.weight_decay
        )
        decoder_groups = get_parameter_groups(
            self.decoder, 
            lr=self.cfg.fine_tunning_lr, 
            weight_decay=self.cfg.decoder_weight_decay
        )
        
        # Metric FC is frozen, so groups will be empty, which is fine
        metric_groups = get_parameter_groups(
            self.metric_fc, 
            lr=self.cfg.fine_tunning_lr, 
            weight_decay=self.cfg.weight_decay
        )

        all_groups = encoder_groups + decoder_groups + metric_groups
        optimizer = torch.optim.AdamW(all_groups)
        return optimizer

    def fit(self):
        self.encoder.train()
        self.decoder.train()
        self.metric_fc.eval() 

        # --- CRITICAL FIX: Freeze BN Statistics ---
        # We force BatchNorm layers to eval mode so they use the saved running stats
        # instead of the current batch stats. This prevents the "Loss Explosion".
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
        for module in self.decoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                
        total_loss = 0.0
        total_recon_loss = 0.0
        total_cls_loss = 0.0
        
        correct = 0
        total_samples = 0

        for batch in self.train_loader:
            img, label = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            # 1. Forward Pass
            embeddings = self.encoder(img)
            reconstruction = self.decoder(embeddings)
            
            # 2. Loss Calculation
            # Pass label to get penalized logits (ArcFace Margin) -> Good for optimization
            logits_margin = self.metric_fc(embeddings, label)
            loss_cls = self.criterion_cls(logits_margin, label)
            
            loss_recon = self.criterion_recon(reconstruction, img)

            # Combine Losses
            loss = (self.cfg.loss_pixel * loss_recon) + loss_cls

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)
            
            self.optimizer.step()

            # 3. Metrics Calculation
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cls_loss += loss_cls.item()

            # --- FIX: Calculate Accuracy on CLEAN logits (No Margin) ---
            with torch.no_grad():
                # label=None tells ArcFace to return raw Cosine Similarity
                logits_clean = self.metric_fc(embeddings, label=None)
            
            _, predicted = torch.max(logits_clean.data, 1)
            total_samples += label.size(0)
            correct += (predicted == label).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon_loss / len(self.train_loader)
        avg_cls = total_cls_loss / len(self.train_loader)
        acc = 100. * correct / total_samples

        return avg_loss, avg_recon, avg_cls, acc

    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        self.metric_fc.eval()
        
        correct = 0
        total = 0
        total_recon_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img, label = batch[0].to(self.device), batch[1].to(self.device)
                
                embeddings = self.encoder(img)
                recon = self.decoder(embeddings)
                
                # Check Classification Accuracy (No Margin)
                logits = self.metric_fc(embeddings, label=None) 
                _, predicted = torch.max(logits.data, 1)
                
                loss_recon = self.criterion_recon(recon, img)
                
                total += label.size(0)
                correct += (predicted == label).sum().item()
                total_recon_loss += loss_recon.item()

        val_acc = 100. * correct / total
        avg_recon = total_recon_loss / len(self.val_loader)
        
        return val_acc, avg_recon

    def run(self):
        print(f"--- Starting Fine-Tuning (Epochs: {self.cfg.fine_tunning_epochs}) ---")
        
        for epoch in range(self.cfg.fine_tunning_epochs):
            self.current_epoch = epoch
            
            loss, loss_recon, loss_cls, train_acc = self.fit()
            val_acc, _ = self.validate()

            self.writer.add_scalar("FineTune/TotalLoss", loss, epoch)
            self.writer.add_scalar("FineTune/ReconLoss", loss_recon, epoch)
            self.writer.add_scalar("FineTune/ClsLoss", loss_cls, epoch)
            self.writer.add_scalar("FineTune/TrainAcc", train_acc, epoch)
            self.writer.add_scalar("FineTune/ValAcc", val_acc, epoch)

            print(
                f"FT Epoch {epoch:02d} | "
                f"Loss: {loss:.4f} (R:{loss_recon:.3f} C:{loss_cls:.3f}) | "
                f"Tr Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
            )

        torch.save(self.encoder.state_dict(), f"{self.cfg.save_dir}/finetuned_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.cfg.save_dir}/finetuned_decoder.pth")
        print("--- Fine-Tuning Complete ---")