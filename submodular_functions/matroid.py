import numpy as np

# note that all the sets that we are using here are the sets of indices and not the actual elements - this is the standard notation that is usually followed

class PartitionMatroid():
    def __init__(self, V: list, partition_labels: list, limits: list):
        u_labels = set(partition_labels)
        assert len(u_labels) == len(limits) # here we are asserting that the number of unique labels i.e. the number of partitions equals the number of 
        assert min(partition_labels) == 0 # partition labels must follow 0 indexing
        self.V = V
        self.parition_labels = partition_labels
        self.num_paritions = len(limits)
        self.limits = limits
    
    def rank(self, A: set):
        r = 0
        for i in range(0, self.num_paritions):
            V_i = set(list(np.array(self.V)[list(np.nonzero(self.parition_labels == i))]))
            r+=min(len(A.intersection(set(V_i))), self.limits[i])
        return r





