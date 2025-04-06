import torch
import torch.nn as nn
import torch.nn.functional as F
# 偏置cosine sim
class CosineClassifierWithBias(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))  # 偏置

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine_sim = torch.matmul(x_norm, w_norm.T)
        return cosine_sim + self.bias
    def apply_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight)

class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)