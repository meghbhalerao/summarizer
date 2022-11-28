import numpy as np
import os
import torch
from submodular_functions.facility_location import make_kernel
import wandb
import sys
from greedy import greedy_max
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.featurize import featurize_data
sns.set_theme()

def main():
    #wandb.init()
    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu == False:
        raise ValueError("Can only proceed if gpu is available!")

    data_set  = "20newsgroups"
    feat_type = 'tfidf'
    df_path = os.path.join("./saved_stuff/processed_data/", data_set, feat_type, "processed_data.pkl")

    if data_set == '20newsgroups':
        data_path = os.path.join("./downloaded_data/", "20newsgroups_raw.csv")
        dataset = pd.read_csv(data_path)
        df = featurize_data(dataset, dname = data_set, data_path = None, df_path = df_path)
        print(len(df['feature']))

    elif data_set == "airbnb":
        data_path = os.path.join("./downloaded_data/", "airbnb_data/images/")  
        df = featurize_data(pd.DataFrame(), dname = data_set, data_path = data_path, df_path=df_path)
    else:
        raise ValueError(f"Dataset {data_set} entered! Not yet supported!")

    # sanity check to find the rank of the entire dataset and that must be equal to the number of classes according to our partition matroid rank function
    mat = PartitionMatroid()

    if use_gpu:
        dataset = torch.tensor(dataset).cuda()
    print("shape of the dataset is", dataset.shape)
    n_data = dataset.shape[0]
    print(f"number of data points are {n_data}")
    W  = make_kernel(dataset)

    print("shape of the symmetric similarity kernel is ", W.shape)

    # doing some initializations
    V = set(list(np.arange(0, n_data)))

    greedy_max(dataset, V, k = 10, fn = 'FL', W = W, greedy_type = 'standard')

if __name__ == '__main__':
    main()


def plot_points(dataset, data_name):
    fig, ax  = plt.subplot()
    ax.scatter(dataset[:,0], dataset[:,1], marker='x')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    fig.savefig("./figs/x.png")
    wandb.log({"img": wandb.Image("./figs/x.png")})