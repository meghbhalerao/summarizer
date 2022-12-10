import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch 
import random
def get_random_stats(mat, V, k, num_trials = 100, max_rank = None):  
    random_rank_list = []
    for _ in range(num_trials):
        random_idxs = set(random.sample(V, k))
        random_rank_list.append(int(mat.rank(random_idxs)))
    mean = np.mean(random_rank_list)
    stddev = np.std(random_rank_list)
    print(f"mean is {mean} and std dev is {stddev}")
    bins = np.arange(0, max_rank + 1.5) - 0.5
    #bins = np.arange(0, max_rank)
    fig, ax = plt.subplots()
    plt.draw()

    fig.set_tight_layout(True)

    _ = ax.hist(np.array(random_rank_list).astype(int), bins)
    ax.set_xticks(bins)
    ax.set_xlim([230, 270])
    ax.set_xlabel('rank values')
    ax.set_ylabel('set count')
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xticklabels(ax.get_xticks(), rotation = 90)
    plt.savefig("x.png")

def do_clustering(n_clusters, feature_matrix,algo = 'k-means', init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=1, random_state=0, copy_x=True, algorithm='auto'):
    if algo == 'k-means':
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=1, tol=0.0001, verbose=verbose, random_state=random_state, copy_x=True, algorithm='auto')
    else:
        raise NotImplementedError(f"Clustering algo {algo} not implemented yet!")

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =  True
