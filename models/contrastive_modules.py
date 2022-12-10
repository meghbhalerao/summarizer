import torch.nn as nn
import lightly.models as models

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = models.modules.SimCLRProjectionHead(
          input_dim=512, 
          hidden_dim=512,
          output_dim=128
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
