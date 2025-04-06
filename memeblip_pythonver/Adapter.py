from torch import nn
import torch

# Adapter类定义（优化版）
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=1.5, dropout_rate=0.1):
        super(Adapter, self).__init__()
        reduced_dim = max(16, int(c_in // reduction))
        self.norm1 = nn.LayerNorm(c_in)
        self.fc = nn.Sequential(
            nn.Linear(c_in, reduced_dim, bias=False),
            nn.GELU(),
            nn.Linear(reduced_dim, c_in, bias=False)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(c_in)
        self.scale = nn.Parameter(torch.tensor(0.1))  # 残差缩放系数
        self.apply(self.init_weights)

    def forward(self, x):
        residual = x
        x = self.fc(self.norm1(x))
        x = self.dropout(x)
        x = residual + self.scale * x
        return self.norm2(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')


def monitor_adapter_training(adapter, input_data, target, criterion):
    adapter.train()
    
    output = adapter(input_data)
    loss = criterion(output, target)

    print("\n===== Adapter训练状态监控 =====")
    print(f"Adapter Loss: {loss.item():.6f}")

    # 梯度只在Lightning的主训练step中进行，这里只做前向，不调用backward
    # 通过Lightning的主训练过程计算的梯度来查看
    for name, param in adapter.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"梯度 {name}: mean={grad_mean:.6f}, std={grad_std:.6f}")
        else:
            print(f"梯度 {name}: 无梯度")
