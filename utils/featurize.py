import pandas as pd
import texthero as hero
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage import io
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel, ViTConfig, ViTModel, ViTFeatureExtractor, AutoFeatureExtractor, ResNetModel, BeitFeatureExtractor, BeitForImageClassification
from models.networks import *
import torch 
import sys
from tqdm import tqdm
import torch.nn as nn
from dataset import AirBnbDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import *
from models.contrastive_modules import MoCo
sys.path.append("../")
from models.contrastive_modules import SimCLR

def featurize_data(df: pd.DataFrame, dname = '20newsgroups', feat_type = None, data_path = None, df_path = None, calculate_stuff = None, feat_contrastive_algo = None, feat_contrastive_model = None, sbert_model_name = None):
    assert calculate_stuff is not None
    if os.path.exists(df_path) and not calculate_stuff:
        print(f"found existing processed data at {df_path}")
        return pickle.load(open(df_path, 'rb'))
    else:
        pass
    
    if dname == '20newsgroups':
        df['raw_text'] = hero.clean(df['raw_text'])
        print(df['raw_text'])
        le = preprocessing.LabelEncoder()
        le.fit(df.label)
        df['categorical_label'] = le.transform(df.label)

        if feat_type == 'tfidf':
            path_emb = os.path.join("saved_stuff/saved_embeddings/", dname, feat_type)
            if os.path.exists(path_emb) and not calculate_stuff:
                embeddings = pickle.load(open(os.path.join(path_emb, "embeddings.pkl"), 'rb'))
            else:
                os.makedirs(path_emb, exist_ok=True)
                embeddings = hero.tfidf(df['raw_text'], max_features=100)
                pickle.dump(embeddings, open(os.path.join(path_emb, "embeddings.pkl"), 'wb'))

            embeddings = np.nan_to_num(np.array(embeddings))
            df['feature'] = list(np.array(embeddings))


        elif feat_type == 'sbert':
            model = SentenceTransformer(sbert_model_name)
            sentences = list(df['raw_text'])
            path_emb = os.path.join("saved_stuff/saved_embeddings/", dname, feat_type, "embeddings.pkl")
            if os.path.exists(path_emb) and not calculate_stuff:
                embeddings = pickle.load(open(path_emb, 'rb'))
            else:
                embeddings = model.encode(sentences)
                pickle.dump(embeddings, open(path_emb, 'wb'))

            embeddings = np.nan_to_num(np.array(embeddings))
            df['feature'] = list(np.array(embeddings))
        
        else:
            raise ValueError(f"feature type {feat_type} entered for {dname} which is not supported yet!")
            
    elif dname == 'airbnb':
        num_channels = 3
        base_path = data_path
        img_list_temp = os.listdir(base_path)
        img_list = []
        for x in img_list_temp:
            if x[0] != '.' and x[0] != '_':
                img_list.append(x)
        
        df['img_path'] = img_list
        text_label_list = []
        for img_name in img_list:
            text_label_list.append("_".join(str(img_name).split('_')[:-1]))

        df['label'] = text_label_list
        le = preprocessing.LabelEncoder()
        le.fit(df.label)
        df['categorical_label'] = le.transform(df.label)
        feature_list = []
        dst_airbnb = AirBnbDataset(df, base_path)
        mean_list = []
        std_list = []

        images_all = torch.cat([torch.unsqueeze(dst_airbnb[i][0], dim=0) for i in range(len(dst_airbnb))], dim = 0)
        
        for ch in range(num_channels):
            mean_list.append(torch.mean(images_all[:, ch]).item())
            std_list.append(torch.std(images_all[:,ch]).item())
        print(f"mean: {mean_list} and std: {std_list}")
        dst_airbnb = AirBnbDataset(df, base_path, transform = transforms.Normalize(mean_list, std_list))
        dl_airbnb = DataLoader(dst_airbnb, batch_size=1, shuffle = False)

        if feat_type == 'sift':
            descriptor_extractor = SIFT()
            max_descriptors = 200
            for img_path in tqdm(df['img_path']):
                img = np.array(ImageOps.grayscale(Image.open(os.path.join(base_path, str(img_path)))))
                descriptor_extractor.detect_and_extract(img)
                feature_list.append(descriptor_extractor.keypoints.flatten()[:max_descriptors])

                print(len(descriptor_extractor.keypoints.flatten()[:max_descriptors]))
            df['feature'] = feature_list

        elif feat_type == 'resnet50-imagenet':
            layer_feat = 'avgpool'
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
   
            im_size = (224,224)

            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list

        elif feat_type == 'resnet34-imagenet':
            layer_feat = 'avgpool'
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
        
        elif feat_type == 'resnet101-imagenet':
            layer_feat = 'avgpool'
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

            feature_list = extract_feature_nn(model, layer_feat = layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
        
        elif feat_type == 'resnet152-imagenet':
            layer_feat = 'avgpool'
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list   

        elif feat_type == 'resnext50_32x4d-imagenet':
            layer_feat = 'avgpool'
            model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

            feature_list = extract_feature_nn(model, layer_feat = layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list  
    
        elif feat_type == 'resnext101_32x8d-imagenet':
            layer_feat = 'avgpool'
            model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

            feature_list = extract_feature_nn(model, layer_feat = layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
        
        elif feat_type == 'resnext101_64x4d-imagenet':
            layer_feat = 'avgpool'
            model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
            print(model)
            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
        
        elif feat_type == 'convnext_tiny':
            layer_feat = 'avgpool'
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            print(model)
            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
        
        elif feat_type == 'convnext_small':
            layer_feat = 'avgpool'
            model = convnext_small(weights='DEFAULT')

            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list

        elif feat_type == 'convnext_base':
            layer_feat = 'avgpool'
            model = convnext_base(weights='DEFAULT')
            print(model)
            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list

        elif feat_type == 'convnext_large':
            layer_feat = 'avgpool'
            model = convnext_large(weights='DEFAULT')

            feature_list = extract_feature_nn(model, layer_feat= layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list

        elif feat_type == 'contrastive':
            if feat_contrastive_model == 'resnet18':
                resnet = torchvision.models.resnet18()
                backbone = nn.Sequential(*list(resnet.children())[:-1])
                if feat_contrastive_algo == 'simclr':
                    model  = SimCLR(backbone)
                elif feat_contrastive_algo == 'moco':
                    model = MoCo(backbone)
                model.load_state_dict(torch.load(os.path.join(f"saved_stuff/saved_models/{feat_contrastive_model}_{feat_contrastive_algo}_best_model.pt")))
                model = model.backbone
                model.eval()
                
           
            else:
                raise NotImplementedError()
            feature_list = extract_feature_nn(model, layer_feat= None, dl = dl_airbnb)
            df['feature'] = feature_list
            
        elif feat_type == 'resnet18-imagenet':
            layer_feat = 'avgpool'
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            
            im_size = (224,224)
            feature_list = extract_feature_nn(model, layer_feat=layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list


        elif feat_type == 'random-convnet':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
            layer_feat = 'features.6'
            im_size = (224,224)
            feature_list = []
            model  = ConvNet(channel=3, num_classes=356, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size, bias = True)

            feature_list = extract_feature_nn(model, layer_feat=layer_feat, dl = dl_airbnb)
            df['feature'] = feature_list
            
        elif feat_type == 'gist':
            pass

        elif feat_type == 'clip':
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            for img_path in df['img_path']:
                img = Image.open(os.path.join(base_path, str(img_path)))   
            raise ValueError("CLIP feature extractor not fully implemented yet")

        elif feat_type == 'vit':
            feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            with torch.no_grad():
                for img_path in tqdm(df['img_path']):
                    img = Image.open(os.path.join(base_path, str(img_path)))
                    inputs = feature_extractor(img, return_tensors="pt")
                    print(inputs.shape)
                    x = model.forward(**inputs , output_hidden_states = True)
         
                    feature_list.append(x.last_hidden_state.view(-1).cpu().detach().numpy())
                
            df['feature'] = feature_list

        elif feat_type == 'resnet':
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            model = ResNetModel.from_pretrained("microsoft/resnet-50")
            with torch.no_grad():
                for img_path in df['img_path']:
                    img = Image.open(os.path.join(base_path, str(img_path)))
                    inputs = feature_extractor(img, return_tensors="pt")
                    feature_list.append(model.forward(inputs, output_hidden_states = True).last_hidden_state.flatten().cpu().detach().numpy())
            
            df['feature'] = feature_list
        elif feat_type == 'biet':
            raise NotImplementedError()
        else:
            raise ValueError(f"feature type {feat_type} entered for {dname} which is not supported yet!")

    pickle.dump(df, open(df_path, 'wb'))

    return df

def extract_feature_nn(model, layer_feat = None, dl = None, verbose = True):
    feature_list = []
    if layer_feat is not None:
        net_feature = FeatureExtractor(model, layers = [layer_feat]).cuda().eval()
    else:
        net_feature = model
    model.cuda()
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(dl)):
            feat_vec = net_feature(img.cuda())[layer_feat].view(-1).cpu().detach().numpy()
            feature_list.append(feat_vec)
            print(feat_vec.shape)
    return feature_list



