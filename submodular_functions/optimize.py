import submodlib

def instantiate_function(fn = 'facility_location', n_data = None, mode = 'dense', sim_kernel = None, k = None, df = None):
    fl_obj = submodlib.functions.facilityLocation.FacilityLocationFunction(n_data, mode, separate_rep=False, n_rep=None, sijs=sim_kernel, data=None, data_rep=None, num_clusters=None, cluster_labels=None, num_neighbors=None, create_dense_cpp_kernel_in_python=True, pybind_mode='array')
    dm_obj = submodlib.functions.disparityMin.DisparityMinFunction(n_data, mode, sijs=sim_kernel, data=None, num_neighbors=None)
    ld_obj = submodlib.functions.logDeterminant.LogDeterminantFunction(n_data, mode, 1e-10, sijs=sim_kernel, data=None, num_neighbors=None)
    feat_dim = df['feature'][0].shape[0]
    feat_list = list(df['feature'])
    print("length of feature list is", len(feat_list))
    fb_obj = submodlib.functions.featureBased.FeatureBasedFunction(n_data, feat_list, feat_dim, sparse = False)

    if fn == 'facility_location':
        return fl_obj
    
    elif fn == 'disparity_min':
        return dm_obj
    
    elif fn == 'log_det':
        return ld_obj

    elif fn == 'feature_based':
        return fb_obj

    elif isinstance(fn, list):
        fn_obj_list = []
        for fn_name in fn:
            if fn_name == 'facility_location':
                fn_obj_list.append(fl_obj)
            if fn_name == 'disparity_min':
                fn_obj_list.append(dm_obj)
            if fn_name == 'log_det':
                fn_obj_list.append(ld_obj)
            if fn_name  == 'feature_based':
                fn_obj_list.append(fb_obj)
        assert len(fn_obj_list) > 0
        return fn_obj_list
    else:
        raise ValueError(f"submodular function {fn} used, but it is not implemented yet!")
        

    