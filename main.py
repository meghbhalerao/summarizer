import numpy as np
import os
import torch
from submodular_functions.facility_location import make_kernel, facility_location
import wandb
from utils.utils import get_random_stats
import sys
from greedy import greedy_max
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.featurize import featurize_data
from submodular_functions.matroid import PartitionMatroid
from submodular_functions.optimize import instantiate_function
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf

sns.set_theme()

@hydra.main(version_base = None, config_path="configs", config_name="parent_config")
def main(config_dict):
    config_dict = OmegaConf.to_container(
        config_dict, resolve=True, throw_on_missing=True
    )
    if config_dict["log_wandb"]:
        wandb.init()

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu == False:
        raise ValueError("Can only proceed if gpu is available!")

    data_set  = config_dict["data_set"]
    feat_type = config_dict["feat_type"]
    submod_function = config_dict["submod_function"]
    compute_random_stats = config_dict["compute_random_stats"]
    distance_metric = config_dict["distance_metric"]
    similarity_kernel = config_dict["similarity_kernel"]
    use_sml = config_dict["use_sml"]
    calculate_stuff = config_dict["calculate_stuff"]
    print("Submodlib library usage status is", use_sml)
    print("config dict for debugging purposes is ", config_dict)
    if data_set == '20newsgroups':
        k = 20
    elif data_set == 'airbnb':
        k = 356

    base_exp_path = os.path.join("./saved_stuff/processed_data/", data_set, feat_type)
    os.makedirs(base_exp_path, exist_ok=True)
    df_path = os.path.join(base_exp_path, "processed_data.pkl")

    if data_set == '20newsgroups':
        data_path = os.path.join("./downloaded_data/", "20newsgroups_raw.csv")
        dataset = pd.read_csv(data_path)
        dataset["raw_text"].replace('', np.nan, inplace=True)
        dataset = dataset.dropna(subset=['raw_text'])
        dataset = dataset.dropna().reset_index(drop=True)
        df = featurize_data(dataset, dname = data_set, feat_type=feat_type, data_path = None, df_path = df_path, calculate_stuff=calculate_stuff)

    elif data_set == "airbnb":
        data_path = os.path.join("./downloaded_data/", "airbnb_data/images/")  
        df = featurize_data(pd.DataFrame(), dname = data_set, feat_type=feat_type, data_path = data_path, df_path = df_path, calculate_stuff = calculate_stuff)
    else:
        raise ValueError(f"Dataset {data_set} entered! Not yet supported!")

    feat_vec = np.array(list(df['feature'])).astype(float)
    
    if use_gpu:
        feat_vec = torch.tensor(feat_vec).cuda()

    print("shape of the features of dataset is", feat_vec.shape)
    n_data = feat_vec.shape[0]
    print(f"number of data points are {n_data}")
    # sanity check to find the rank of the entire dataset and that must be equal to the number of classes according to our partition matroid rank function

    V = list(np.arange(n_data))
    partition_labels = df['categorical_label']
    print(partition_labels)
    limits = list(np.ones(len(set(partition_labels))))
    mat = PartitionMatroid(V, partition_labels, limits)
    full_matroid_rank = mat.rank(set(V))

    print("rank of full matroid is", full_matroid_rank)
    if compute_random_stats:
        get_random_stats(mat, V, k, num_trials=10000, max_rank = full_matroid_rank)
        sys.exit()

    kernel_path = os.path.join(base_exp_path, "kernel", distance_metric, similarity_kernel)
    if os.path.exists(kernel_path) and not calculate_stuff:
        print(f"similarity kernel exists at {kernel_path}, loading it!")
        W = pickle.load(open(os.path.join(kernel_path, "kernel.pkl"), 'rb'))
    else:
        print("calculating similarity kernel ...")
        W  = make_kernel(feat_vec, metric=distance_metric, similarity=similarity_kernel)
        os.makedirs(kernel_path,exist_ok=True)
        pickle.dump(W, open(os.path.join(kernel_path, "kernel.pkl"), 'wb'))

    if use_sml:
        W = np.array(W.cpu())
        function_obj = instantiate_function(fn = submod_function, n_data = n_data, sim_kernel = W, k = k, df = df)
    else:
        if submod_function == 'facility_location':
            function_obj = facility_location
        elif submod_function == 'clustering':
            do_clustering(n_clusters = k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=1, random_state=seed, copy_x=True, algorithm='auto')
            kmeans = KMeans(n_clusters=n_clusters, )
        else:
            raise ValueError(f"custom implmentation of {submod_function} not supported yet!")
        #A_max = function_obj.maximize(k)
        #A_set = set([x[0] for x in A_max])
        #print(A_set)
        #print(mat.rank(A_set))
    

        
    print("shape of the symmetric similarity kernel is ", W.shape)

    x = greedy_max(feat_vec, V, k = k, function_obj = function_obj, fn_name = 'FL', W = W, greedy_type = 'standard', sml = use_sml)
    print(mat.rank(x))

if __name__ == '__main__':
    main()


def plot_points(dataset, data_name):
    fig, ax  = plt.subplot()
    ax.scatter(dataset[:,0], dataset[:,1], marker='x')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    fig.savefig("./figs/x.png")
    wandb.log({"img": wandb.Image("./figs/x.png")})