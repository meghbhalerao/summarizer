import torch
import os
import random
from torch.utils.data import Subset
from PIL import Image
import pickle
import sys
sys.path.append("/x0/megh98/projects/CAL/")
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def permute_channels(tsr):
    print(tsr.shape)
    perm = torch.LongTensor([2,1,0])
    tsr = tsr[perm, :, :]
    return tsr
    
def image_grid(imgs, rows, cols, original = False):
    print(rows, cols, len(imgs))
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        if original:
            img = img.convert("RGB")
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

datasets_path = os.path.join("../al_samples_round/glister/")
save_path = os.path.join("imgs_montage", "glister")
al_method = "glister"
rnd = False
tensor_to_pil = T.ToPILImage()
n_samples = 100
n_rows = 10
n_cols = 10
dl_path = datasets_path
datasets = []
for dl in sorted(os.listdir(dl_path)):
    dl = dl.replace("_unnorm","")
    datasets.append(dl)
datasets = list(set(datasets))
derma_all_random = []
path_all_random = []
derma_all_entropy = []
path_all_entropy = []

derma_all_unnorm = []
path_all_unnorm = []


def show_dataset(data_set, data_set_unnorm, dl_name,  n_samples = 100, n_rows = 10, n_cols = 10):
    n_loader = len(data_set)
    idxs  = random.sample(range(0, n_loader), n_samples)
    dset_toshow = Subset(data_set, idxs)
    dset_unnorm_toshow = Subset(data_set_unnorm, idxs)
    img_list = []
    img_list_unnorm = []

    for img, label in dset_toshow:
        pil_img = tensor_to_pil(img)
        img_list.append(pil_img)

    for img, label in dset_unnorm_toshow:
        pil_img = tensor_to_pil(img)
        img_list_unnorm.append(pil_img)

    grid = image_grid(img_list, n_rows, n_cols)
    grid_unnorm = image_grid(img_list_unnorm, n_rows, n_cols, original = True)

    grid.save(os.path.join(save_path, dl_name + ".png"),"PNG")
    grid_unnorm.save(os.path.join(save_path, dl_name + "_unnorm.png"), "PNG")

for idx, dl in enumerate(sorted(datasets)):
    label_list = []
    dset_path = os.path.join(dl_path, dl)
    dl_unnorm = dl.replace(".pt","") + "_unnorm.pt"
    dset_unnorm = os.path.join(dl_path, dl_unnorm)
    print("dset path is", dset_path)
    print("dset unnorm path is", dset_unnorm)
    dset = torch.load(os.path.join(dl_path, dl))
    dset_unnorm = torch.load(os.path.join(dl_path, dl_unnorm))
    print("length of the dataset is", len(dset))
    n_loader = len(dset)
    assert n_loader >=n_samples
    idxs  = random.sample(range(0, n_loader), n_samples)
    dset_toshow = Subset(dset, idxs)
    dset_unnorm_toshow = Subset(dset_unnorm, idxs)
    img_list = []
    img_list_unnorm = []
    for img, label in dset_toshow:
        img = permute_channels(img)
        pil_img = tensor_to_pil(img)
        img_list.append(pil_img)


    for img, label in dset_unnorm_toshow:
        img = permute_channels(img)
        pil_img = tensor_to_pil(img)
        img_list_unnorm.append(pil_img)
    
    
    # plot histogram and save
    for img, label in dset:
        label_list.append(label)

    label_list = np.array(label_list)
    plt.figure()
    sns.histplot(label_list)
    #histfig = histplot.get_figure()
    plt.savefig(os.path.join(save_path, 'histplot_%s.png'%(dl)))
    plt.clf()

    grid = image_grid(img_list, n_rows, n_cols)
    grid_unnorm = image_grid(img_list_unnorm, n_rows, n_cols, original = True)
    grid.save(os.path.join(save_path, dl + ".png"),"PNG")
    grid_unnorm.save(os.path.join(save_path, dl_unnorm + ".png"), "PNG")
    
    if "derma" in dl:
        if "random" in dl:
            derma_all_random.append(dset)
        elif al_method in dl:
            derma_all_entropy.append(dset)
        derma_all_unnorm.append(dset_unnorm)

    elif "path" in dl:
        if "random" in dl:
            path_all_random.append(dset)
        elif al_method in dl:
            path_all_entropy.append(dset)

        path_all_unnorm.append(dset_unnorm)

    #if idx == 10:
    #    break
if rnd: 
    path_full_random = ConcatDataset(path_all_random)
path_full_entropy = ConcatDataset(path_all_entropy)

path_full_unnorm = ConcatDataset(path_all_unnorm)
if rnd:
    derma_full_random = ConcatDataset(derma_all_random)
derma_full_entropy = ConcatDataset(derma_all_entropy)

derma_full_unnorm = ConcatDataset(derma_all_unnorm)

if rnd:
    show_dataset(path_full_random, path_full_unnorm, dl_name = "pathfullrandom",  n_samples = 100)
    show_dataset(derma_full_random, derma_full_unnorm, dl_name = "dermafullrandom",  n_samples = 100)

show_dataset(path_full_entropy, path_full_unnorm, dl_name = "pathfullentropy",  n_samples = 100)
show_dataset(derma_full_entropy, derma_full_unnorm, dl_name = "dermafullentropy",  n_samples = 100)
