
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):

    #the loss should be around ln(total_classes) at the start of training so ln(200)=5.3
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scale factor (Temperature)
        self.m = m  # Angular Margin (The "Gap" between classes)
        
        # These WEIGHTS are your "Prototype Memory"
        # We don't use bias because we want pure direction (angles)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # 1. Normalize Prototypes (Weight) and Input (Embedding)
        # This puts everything on the Sphere (Radius = 1)
        embedding_norm = F.normalize(embedding , p=2 , dim=1)
        weight_norm = F.normalize(self.weight , p=2 ,dim=1)
        cosine = F.linear(embedding_norm, weight_norm)
        if label is None:
            return cosine * self.s
        
        cosine = cosine.clamp(-0.9998, 0.9998)


        if self.m == 0.0:
            return cosine * self.s

        
        # 2. Add the Margin (The "Force Field" that pushes classes apart)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Stability checks (keep math happy)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 3. Create One-Hot view to apply margin ONLY to the correct class
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to correct class, leave others alone
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 4. Scale it up (Temperature)
        output *= self.s
        return output
    
    def set_margin(self, new_m):
        """Updates margin and re-calculates the necessary math constants."""
        self.m = new_m
        self.cos_m = math.cos(new_m)
        self.sin_m = math.sin(new_m)
        self.th = math.cos(math.pi - new_m)
        self.mm = math.sin(math.pi - new_m) * new_m
        print(f"ArcFace Margin updated to: {new_m:.3f}")