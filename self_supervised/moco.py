import torch
from torch import nn
import torchvision
import copy
import lightly.data as data
import os
from lightly.data import LightlyDataset
from lightly.data import MoCoCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
import sys
sys.path.append("../")
from models.contrastive_modules import MoCo


contrastive_algo = 'moco'
backbone_model = 'resnet18'
pretrained =  True
if pretrained:
    resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
else:
    resnet = torchvision.models.resnet18()
    
backbone = nn.Sequential(*list(resnet.children())[:-1])

if contrastive_algo == 'moco':
    model = MoCo(backbone)
else:
    raise NotImplementedError(f"contrastive algorithm {contrastive_algo} not implemented yet!")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = data.LightlyDataset(input_dir=os.path.join("../", "downloaded_data/airbnb_data/images/"))
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MoCoCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=864
    ,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=4096)
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

ckpt_path = os.path.join("../saved_stuff", "saved_models", f"{backbone_model}_{contrastive_algo}_best_model.pt")
print("Starting Training")
best_train_loss = float('inf')
max_epochs = 100

for epoch in range(max_epochs):
    total_loss = 0
    for (x_query, x_key), _, _ in dataloader:
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    if  avg_loss < best_train_loss:
        best_train_loss = avg_loss
        torch.save(model.state_dict(), ckpt_path)
