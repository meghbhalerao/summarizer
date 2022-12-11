import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
import torch 
import random
import scipy

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

def do_clustering(n_clusters, feature_matrix, algo = 'k-means', init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=1, random_state=0, copy_x=True, algorithm='auto', sim_kernel = None):
    if algo == 'k-means':
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        kmeans.fit(feature_matrix)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        chosen_idxs = []
        for i in range(n_clusters):
            cluster_idxs = list(np.nonzero((labels==i).astype(int))[0])
            print(cluster_idxs)
            print(feature_matrix[cluster_idxs, :].shape)
            print(np.expand_dims(centers[i,:], axis = 0).shape)
            min_pt_idx = np.argmin(scipy.spatial.distance.cdist(np.expand_dims(centers[i,:], axis = 0), feature_matrix[cluster_idxs, :]))
            chosen_idxs.append(cluster_idxs[min_pt_idx])
        return set(chosen_idxs)
    
    elif algo == 'spectral':
        sim_kernel = sim_kernel.cpu().numpy()
        clustering = SpectralClustering(n_clusters = n_clusters, affinity='precomputed', n_init=1, assign_labels='cluster_qr', verbose = True, n_jobs = 4, eigen_tol = 1e-10).fit(sim_kernel)
        labels = clustering.labels_

        chosen_idxs = []
        for i in range(n_clusters):
            print(i)
            print((labels==i).sum())
            print(np.nonzero((labels==i).astype(int)))
            cluster_idxs = list(np.nonzero((labels==i).astype(int))[0])
            sim_kernel_intra_cluster = sim_kernel[cluster_idxs,:][:, cluster_idxs]
            cluster_rep = np.argmax(np.sum(sim_kernel_intra_cluster, axis =1))
            chosen_idxs.append(cluster_idxs[cluster_rep])
        return set(chosen_idxs)
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
