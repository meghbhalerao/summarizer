from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
import torch
import sys
from torchmetrics.functional import pairwise_cosine_similarity

def make_kernel(data, metric = 'euclidean', similarity = 'gaussian', sigma = 10):
    if metric == 'euclidean':
        W = torch.cdist(data, data, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    elif metric == "arbitrary":
        pass
    else:
        raise ValueError(f"Entered {metric} metric, it is not supported yet!")

    # check which similarity to use
    if similarity == 'gaussian':
        W = torch.exp(-torch.pow(W,2)/sigma)
    elif similarity == 'linear':
        W = -W
    elif similarity == 'cosine':
        W = pairwise_cosine_similarity(data, data)
    elif similarity == 'dot_product':
        W = data @ data.T
    else:
        raise ValueError(f"Entered {similarity} similarity metric, it is not supported yet!")
    return W

def facility_location(V, A, W):
    """
    A - list of indices of the items in the ground set V
    V - the entire ground set V, containing all the indices that we want
    W - the similarity matrix of size n x n, where n is the cardinality of V
    """
    if isinstance(A, set): # better to use standard list indexing rather than set indexing
        A = list(A)

    W_A = W[:,A] # this is the submatrix of the entire symmetric matrix indexed by A on the columns of the W matrix - this means that the rows index the elements of V and the columns index the elements of A

    if len(A) == 0:
        fl_val = 0
    else:
        fl_val = torch.sum(torch.max(W_A, dim = 1).values)
    return fl_val

