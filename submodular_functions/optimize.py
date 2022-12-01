import submodlib

def instantiate_function(fn = 'facility_location', n_data = None, fl_mode = 'dense', sim_kernel = None, k = None):
    if fn == 'facility_location':
        f = submodlib.functions.facilityLocation.FacilityLocationFunction(n_data, fl_mode, separate_rep=False, n_rep=None, sijs=sim_kernel, data=None, data_rep=None, num_clusters=None, cluster_labels=None, metric='cosine', num_neighbors=None, create_dense_cpp_kernel_in_python=True, pybind_mode='array')
        return f
        

    