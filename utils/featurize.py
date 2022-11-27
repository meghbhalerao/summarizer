import pandas as pd
import texthero as hero
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

def featurize_data(df: pd.DataFrame, dname = '20newsgroups', feat_type = 'tfidf', data_path = None):
    df_path = os.path.join("../saved_stuff/processed_data/")
    if dname == '20newsgroups':
        if feat_type == 'tfidf':
            df['raw_text'] = hero.clean(df['raw_text'])
            le = preprocessing.LabelEncoder()
            le.fit(df.label)
            df['categorical_label'] = le.transform(df.label)
            df['feature'] = hero.tfidf(df['raw_text'])

        elif feat_type == 'sbert':
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = list(df['raw_text'])
            path_emb = os.path.join(".../saved_stuff/saved_embeddings/", dname, feat_type, "embeddings.pkl")
            if os.path.exists(path_emb):
                embeddings = pickle.load(open(path_emb, 'wb'))
            else:
                embeddings = model.encode(sentences)
                pickle.dump(embeddings, open(path_emb, 'wb'))

            df['feature'] = list(np.array(embeddings))
            
        
    elif dname == 'airbnb':
        img_list = os.listdir(os.path.join("../downloaded_data/airbnb"))







