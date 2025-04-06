from torch import nn

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super(LinearProjection, self).__init__()

        if isinstance(drop_probs, list):
            dropout_prob = drop_probs[0]
        else:
            dropout_prob = drop_probs

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            layer = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob)
            )
            self.layers.append(layer)

    def forward(self, x):
        # 首次投影
        x = self.input_projection(x)

        # 每个中间层使用残差链接
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # 残差链接

        return x