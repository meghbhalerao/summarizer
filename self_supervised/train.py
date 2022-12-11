import torch
import torchvision
import lightly.models as models
import lightly.loss as loss
import lightly.data as data
import torch.nn as nn
import os
import sys
sys.path.append("../")
from models.contrastive_modules import SimCLR
from utils.utils import set_random_seed
import time

# the collate function applies random transforms to the input images
set_random_seed(0)
collate_fn = data.ImageCollateFunction(input_size=32, cj_prob=0.5)
contrastive_algo = 'simclr'
backbone_model = 'resnet18'
# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir=os.path.join("../", "downloaded_data/airbnb_data/images/"))

# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=1000,         # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn)  # apply transformations to the input images


# use a resnet backbone
if backbone_model == 'resnet18':
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    backbone = nn.Sequential(*list(resnet.children())[:-1])
else:
    raise NotImplementedError(f"backbone model {backbone_model} not implemented yet!")

# build the simclr model
if contrastive_algo == 'simclr':
    model = SimCLR(backbone)
else:
    raise NotImplementedError(f"contrastive algorithm {contrastive_algo} not implemented yet!")

ckpt_path = os.path.join("../saved_stuff", "saved_models", f"{backbone_model}_{contrastive_algo}_best_model.pt")
# lightly exposes building blocks such as loss functions
criterion = loss.NTXentLoss(temperature=0.5)

# get a PyTorch optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_epochs = 1000
model = model.to(device)
best_train_loss = float('inf')

for epoch in range(max_epochs):
    loss_it = 0
    it_count  = 0
    ep_start_time = time.time()
    for (x0, x1), _, _ in dataloader:

        x0 = x0.to(device)
        x1 = x1.to(device)

        z0 = model(x0)
        z1 = model(x1)

        loss_ = criterion(z0, z1)
        loss_.backward()

        optimizer.step()
        optimizer.zero_grad()
        loss_ = loss_.cpu().item()
        if  loss_ < best_train_loss:
            best_train_loss = loss_
            torch.save(model.state_dict(), ckpt_path)
        it_count+=1
        loss_it+=loss_
    ep_end_time = time.time()
    loss_ep = loss_it/it_count
    print(f"epoch loss: {loss_ep}")
    print("time taken for epoch: ", (ep_end_time - ep_start_time)/60, "mins")
    print()


