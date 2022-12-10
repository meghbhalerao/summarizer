import submodlib

def instantiate_function(fn = 'facility_location', n_data = None, mode = 'dense', sim_kernel = None, k = None, df = None):
    if fn == 'facility_location':
        f = submodlib.functions.facilityLocation.FacilityLocationFunction(n_data, mode, separate_rep=False, n_rep=None, sijs=sim_kernel, data=None, data_rep=None, num_clusters=None, cluster_labels=None, num_neighbors=None, create_dense_cpp_kernel_in_python=True, pybind_mode='array')
        return f
    
    elif fn == 'disparity_min':
        f = submodlib.functions.disparityMin.DisparityMinFunction(n_data, mode, sijs=sim_kernel, data=None, num_neighbors=None)
        return f
    
    elif fn == 'log_det':
        f = submodlib.functions.logDeterminant.LogDeterminantFunction(n_data, mode, 1e-10, sijs=sim_kernel, data=None, num_neighbors=None)
        return f

    elif fn == 'feature_based':
        assert df is not None
        feat_dim = df['feature'][0].shape[0]
        print("feature dimension for feature based function is", feat_dim)
        feat_list = list(df['feature'])
        print("length of feature list is", len(feat_list))
        f = submodlib.functions.featureBased.FeatureBasedFunction(n_data, feat_list, feat_dim, sparse = False)
        return f

    elif isinstance(fn, list):
        pass
    else:
        raise ValueError(f"submodular function {fn} used, but it is not implemented yet!")
        

    