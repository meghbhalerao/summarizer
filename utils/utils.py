import random
import numpy as np
import matplotlib.pyplot as plt


def get_random_stats(mat, V, k, num_trials = 100, max_rank = None):  
    random_rank_list = []
    for _ in range(num_trials):
        random_idxs = set(random.sample(V, k))
        random_rank_list.append(int(mat.rank(random_idxs)))
    mean = np.mean(random_rank_list)
    stddev = np.std(random_rank_list)
    print(f"mean is {mean} and std dev is {stddev}")
    bins = np.arange(0, max_rank + 1.5) - 0.5
    fig, ax = plt.subplots()
    _ = ax.hist(random_rank_list, bins)
    ax.set_xticks(bins + 0.5)
    ax.set_xlabel('rank values')
    ax.set_ylabel('set count')
    plt.savefig("x.png")
