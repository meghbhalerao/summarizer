CUDA_VISIBLE_DEVICES=3 python main.py data_set=airbnb feat_type=contrastive feat_contrastive_algo=moco feat_contrastive_model=resnet18 calculate_stuff=true submod_function=facility_location distance_metric=arbitrary similarity_kernel=cosine sigma=0.1
