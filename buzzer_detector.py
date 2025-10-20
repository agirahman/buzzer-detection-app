import pandas as pd
import numpy as np
import json
import re
import networkx as nx
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle
import os

class BuzzerDetector:
    def __init__(self):
        self.model = None
        self.features = [
            'pagerank', 'betweenness', 'in_degree', 'out_degree',
            'narrative_similarity', 'tweet_frequency', 'reply_ratio'
        ]

    def preprocess_json_data(self, json_data):
        """Phase 1: Preprocessing data JSON menjadi master_dataset_cleaned.csv"""
        all_tweets = []

        # Handle jika json_data adalah list atau dict
        if isinstance(json_data, list):
            data_list = json_data
        else:
            data_list = [json_data]

        for data in data_list:
            if 'tweets' in data:
                tweets = data['tweets']
            elif isinstance(data, list):
                tweets = data
            else:
                tweets = [data]

            for tweet in tweets:
                # Ekstrak fields yang diperlukan
                tweet_id = tweet.get('tweetId', '')
                user_name = tweet.get('userName', '')
                content = tweet.get('content', '')
                like_count = tweet.get('likeCount', 0)
                retweet_count = tweet.get('retweetCount', 0)
                reply_count = tweet.get('replyCount', 0)
                quote_count = tweet.get('quoteCount', 0)
                created_at = tweet.get('createdAt', '')

                # Ekstrak reply_to_user dari content
                reply_to_user = self.extract_reply_to(content)

                # Bersihkan teks
                cleaned_text = self.clean_text(content)

                all_tweets.append({
                    'tweet_id': tweet_id,
                    'username': user_name,
                    'text': content,
                    'cleaned_text': cleaned_text,
                    'reply_to_user': reply_to_user,
                    'like_count': like_count,
                    'retweet_count': retweet_count,
                    'reply_count': reply_count,
                    'quote_count': quote_count,
                    'created_at': created_at
                })

        df = pd.DataFrame(all_tweets)
        return df

    def extract_reply_to(self, text):
        """Ekstrak username yang di-reply dari teks"""
        if not isinstance(text, str):
            return None
        # Cari pola @username di awal tweet
        match = re.match(r'@(\w+)', text.strip())
        if match:
            return match.group(1)
        return None

    def clean_text(self, text):
        """Bersihkan teks: hapus URL, mention, hashtag, karakter non-alfabet"""
        if not isinstance(text, str):
            return ''
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Hapus mention
        text = re.sub(r'@\w+', '', text)
        # Hapus hashtag
        text = re.sub(r'#\w+', '', text)
        # Hapus karakter non-alfabet kecuali spasi
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Lowercase dan strip
        text = text.lower().strip()
        return text

    def build_graph_and_sna(self, df_master):
        """Phase 2: Build graph dan ekstrak fitur SNA"""
        G = nx.DiGraph()

        # Filter interaksi reply yang valid
        interaction_df = df_master.dropna(subset=['reply_to_user'])

        # Tambah edges
        for _, row in interaction_df.iterrows():
            source = row['username']
            target = row['reply_to_user']
            G.add_edge(source, target)

        if G.number_of_nodes() == 0:
            # Jika tidak ada interaksi, buat node kosong dengan fitur default
            df_nodes = pd.DataFrame({
                'username': df_master['username'].unique(),
                'pagerank': 0.0,
                'betweenness': 0.0,
                'in_degree': 0,
                'out_degree': 0,
                'community': 0
            })
            return df_nodes

        # Hitung metrik SNA
        in_degree_dict = dict(G.in_degree())
        out_degree_dict = dict(G.out_degree())
        pagerank_dict = nx.pagerank(G, alpha=0.85)
        betweenness_dict = nx.betweenness_centrality(G)

        # Community detection
        G_undirected = G.to_undirected()
        partition_dict = community_louvain.best_partition(G_undirected)

        # Buat DataFrame
        df_nodes = pd.DataFrame(G.nodes(), columns=['username'])
        df_nodes['pagerank'] = df_nodes['username'].map(pagerank_dict).fillna(0)
        df_nodes['betweenness'] = df_nodes['username'].map(betweenness_dict).fillna(0)
        df_nodes['in_degree'] = df_nodes['username'].map(in_degree_dict).fillna(0)
        df_nodes['out_degree'] = df_nodes['username'].map(out_degree_dict).fillna(0)
        df_nodes['community'] = df_nodes['username'].map(partition_dict).fillna(0)

        return df_nodes

    def feature_engineering(self, df_network, df_master):
        """Phase 3: Feature engineering konten dan perilaku"""
        # Merge data
        df_features = pd.merge(df_network, df_master, on='username', how='inner')

        # Narrative similarity
        df_features['narrative_similarity'] = 0.0
        grouped_by_community = df_features.groupby('community')

        for community_id, group in grouped_by_community:
            if len(group) > 1:
                corpus = group['cleaned_text'].fillna('').tolist()
                vectorizer = TfidfVectorizer(stop_words=['yang', 'di', 'ini', 'itu', 'dan', 'dengan'])
                tfidf_matrix = vectorizer.fit_transform(corpus)
                centroid = np.asarray(tfidf_matrix.mean(axis=0))
                similarity_scores = cosine_similarity(tfidf_matrix, centroid)
                df_features.loc[group.index, 'narrative_similarity'] = similarity_scores.flatten()

        # Behavioral features
        tweet_counts = df_master['username'].value_counts().reset_index()
        tweet_counts.columns = ['username', 'tweet_frequency']
        df_features = pd.merge(df_features, tweet_counts, on='username', how='left')
        df_features['reply_ratio'] = (df_features['out_degree'] / df_features['tweet_frequency']).fillna(0)

        # Select final features
        final_cols = ['username', 'community'] + self.features
        df_final_features = df_features[final_cols].drop_duplicates(subset=['username']).reset_index(drop=True)

        return df_final_features

    def heuristic_labeling(self, df_final):
        """Phase 4: Heuristic labeling"""
        # Hitung thresholds
        narrative_thresh = df_final['narrative_similarity'].quantile(0.75)
        out_degree_thresh = df_final['out_degree'].quantile(0.75)

        def label_buzzer(row):
            is_isolated = (row['in_degree'] == 0) and (row['betweenness'] == 0)
            is_high_activity = (row['narrative_similarity'] > narrative_thresh) and (row['out_degree'] > out_degree_thresh)
            return 1 if is_isolated and is_high_activity else 0

        df_final['is_buzzer'] = df_final.apply(label_buzzer, axis=1)
        return df_final

    def train_model(self, df_labeled):
        """Phase 5: Train model"""
        X = df_labeled[self.features]
        y = df_labeled['is_buzzer']

        if y.sum() == 0:
            # Jika tidak ada buzzer, buat model dummy
            self.model = None
            return

        ratio = y.value_counts()[0] / y.value_counts()[1]
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X, y)
        self.model = model

    def inference(self, df_final):
        """Phase 6: Inference"""
        if self.model is None:
            df_final['buzzer_probability'] = 0.0
        else:
            X = df_final[self.features]
            probabilities = self.model.predict_proba(X)[:, 1]
            df_final['buzzer_probability'] = probabilities

        # Sort by probability
        df_results = df_final.sort_values('buzzer_probability', ascending=False)
        return df_results

    def process_dataset(self, json_data):
        """Main function: Process from raw JSON to buzzer predictions"""
        # Phase 1: Preprocessing
        df_master = self.preprocess_json_data(json_data)

        if df_master.empty:
            return pd.DataFrame()

        # Phase 2: Graph and SNA
        df_network = self.build_graph_and_sna(df_master)

        # Phase 3: Feature Engineering
        df_final = self.feature_engineering(df_network, df_master)

        # Phase 4: Labeling
        df_labeled = self.heuristic_labeling(df_final)

        # Phase 5: Train Model
        self.train_model(df_labeled)

        # Phase 6: Inference
        df_results = self.inference(df_labeled)

        return df_results
