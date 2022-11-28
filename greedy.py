from submodular_functions.facility_location import facility_location
import numpy as np

# greedy for cardinality constrained submodular max - the greedy type can be lazy, random, priority queue greedy, but for now this function only supports the standard greedy algorithm
def get_max_gain_idx(rem, sol, V, W):
    max_gain = -float('inf')
    max_idx = None
    for idx in rem:
        if isinstance(idx, np.int64):
            idx = list([idx])
        idx = set(idx)
        gain = facility_location(V, sol.union(idx), W) - facility_location(V, sol, W)
        if gain > max_gain:
            max_gain = gain
            max_idx = idx
    return set(max_idx)

def greedy_max(V_val, V, k, fn = 'FL', W = None, greedy_type = "standard"):
    if fn == 'FL':
        assert W is not None # the similarity kernel can't be none if we are using a facility location function
    sol = set({})
    while len(sol) < k:
        print("current solution is ", sol)
        rem = V.difference(sol)
        if greedy_type == "standard":
            max_gain_idx = get_max_gain_idx(rem, sol, V, W)
        else:
            raise ValueError(f"entered greedy type {greedy_type}, not supported yet!")
        sol = sol.union(max_gain_idx)





