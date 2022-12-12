import pytorch_lightning as pl
import torchvision
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

# use a resnet backbone
resnet = torchvision.models.resnet18()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
max_epochs = 100

class SimCLR(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = NTXentLoss(temperature=0.5)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-0)
        return optimizer

model = SimCLR(resnet, hidden_dim=512, out_dim=128)
gpus = 1 if torch.cuda.is_available() else None
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(
    model,
    dataloader
)