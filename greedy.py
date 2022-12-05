from submodular_functions.facility_location import facility_location
import numpy as np
from tqdm import tqdm
import torch

# greedy for cardinality constrained submodular max - the greedy type can be lazy, random, priority queue greedy, but for now this function only supports the standard greedy algorithm
def get_max_gain_idx(rem, sol, V, W, function_obj, sml = True):
    max_gain = -float('inf')
    max_idx = None
    for idx in rem:
        if isinstance(idx, np.int64):
            idx = list([idx])
        idx = set(idx)
        if sml == True:
            gain = function_obj.evaluate(sol.union(idx)) - function_obj.evaluate(sol)
        else:
            gain = function_obj(V, sol.union(idx), W) - function_obj(V, sol, W)

        if gain > max_gain:
            max_gain = gain
            max_idx = idx
    return set(max_idx)

def greedy_max(V_val, V, k, function_obj = None ,fn_name = 'FL', W = None, greedy_type = "standard", sml = True):
    # cast W to pytorch 
    assert function_obj is not None
    if not sml:
        W = torch.tensor(W)
    else:
        pass

    if fn_name == 'FL':
        assert W is not None # the similarity kernel can't be none if we are using a facility location function
    sol = set({})
    while len(sol) <= k:
        print("current solution is ", sol)
        rem = set(V).difference(sol)
        if greedy_type == "standard":
            max_gain_idx = get_max_gain_idx(rem, sol, V, W, function_obj, sml = sml) 
        else:
            raise ValueError(f"entered greedy type {greedy_type}, not supported yet!")
        sol = sol.union(max_gain_idx)
    return sol





