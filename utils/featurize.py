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
from transformers import CLIPProcessor, CLIPModel, ViTConfig, ViTModel, ViTFeatureExtractor, AutoFeatureExtractor, ResNetModel
import torch 
import sys

def featurize_data(df: pd.DataFrame, dname = '20newsgroups', feat_type = None, data_path = None, df_path = None):

    if os.path.exists(df_path):
        print(f"found existing processed data at {df_path}")
        return pickle.load(open(df_path, 'rb'))
    else:
        pass
    if dname == '20newsgroups':
        df['raw_text'] = hero.clean(df['raw_text'])
        le = preprocessing.LabelEncoder()
        le.fit(df.label)
        df['categorical_label'] = le.transform(df.label)

        if feat_type == 'tfidf':
            df['feature'] = hero.tfidf(df['raw_text'], max_features=100)

        elif feat_type == 'sbert':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = list(df['raw_text'])
            path_emb = os.path.join(".../saved_stuff/saved_embeddings/", dname, feat_type, "embeddings.pkl")
            if os.path.exists(path_emb):
                embeddings = pickle.load(open(path_emb, 'rb'))
            else:
                embeddings = model.encode(sentences)
                pickle.dump(embeddings, open(path_emb, 'wb'))
            df['feature'] = list(np.array(embeddings))
        
        else:
            raise ValueError(f"feature type {feat_type} entered for {dname} which is not supported yet!")
            
    elif dname == 'airbnb':
        base_path = data_path
        img_list_temp = os.listdir(base_path)
        img_list = []
        for x in img_list_temp:

            if x[0] != '.' and x[0] != '_':
                img_list.append(x)
        
        df['img_path'] = img_list
        text_label_list = []
        for img_name in img_list:
            text_label_list.append("_".join(str(img_name).split('_').pop()))
        
        df['label'] = text_label_list
        le = preprocessing.LabelEncoder()
        le.fit(df.label)
        df['categorical_label'] = le.transform(df.label)
        feature_list = []
        
        if feat_type == 'sift':
            descriptor_extractor = SIFT()
            for img_path in df['img_path']:
                img = np.array(ImageOps.grayscale(Image.open(os.path.join(base_path, str(img_path)))))
                descriptor_extractor.detect_and_extract(img)
                feature_list.append(descriptor_extractor.keypoints.flatten())
                print(len(descriptor_extractor.keypoints.flatten()))
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
                for img_path in df['img_path']:
                    img = Image.open(os.path.join(base_path, str(img_path)))
                    inputs = feature_extractor(img, return_tensors="pt")
    
                    x = model.forward(**inputs , output_hidden_states = True)
                    
                    feature_list.append(x.last_hidden_state.view(-1).cpu().detach().numpy())
                
            df['feature'] = feature_list

        elif feat_type == 'resnet':
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            model = ResNetModel.from_pretrained("microsoft/resnet-50")
            with torch.no_grad():
                for img_path in df['img_path']:
                    img = Image.open(os.path.join(base_path, str(img_path)))
                    feature_list.append(model(img).last_hidden_state.cpu().detach().numpy())
            
            df['feature'] = feature_list

        else:
            raise ValueError(f"feature type {feat_type} entered for {dname} which is not supported yet!")

    pickle.dump(df, open(df_path, 'wb'))

    return df











