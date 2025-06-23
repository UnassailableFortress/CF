# Collaborative Filtering Evaluation Framework (Memory-Efficient Version)

import pandas as pd
import numpy as np
import gzip
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix, csr_matrix
from itertools import combinations
from math import log2
import heapq

# ----------------------------
# Item-to-Item Recommender (SVD-based Approximation)
# ----------------------------
class ItemToItemRecommender:
    def __init__(self, k=10, n_components=100):
        self.k = k
        self.n_components = n_components
        self.item_sim_matrix = None
        self.item_index = None
        self.index_item = None
        self.user_item_matrix = None
        self.user_encoder = None
        self.item_encoder = None
        self.item_factors = None

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col=None):
        ratings = df[rating_col].values if rating_col else np.ones(len(df))

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        user_ids = self.user_encoder.fit_transform(df[user_col])
        item_ids = self.item_encoder.fit_transform(df[item_col])

        self.user_item_matrix = coo_matrix((ratings, (user_ids, item_ids))).tocsr()

        self.item_index = self.item_encoder.classes_
        self.index_item = {i: item for i, item in enumerate(self.item_index)}

        self.item_sim_matrix = self._compute_topk_similar_items_svd(self.user_item_matrix.T)

    def _compute_topk_similar_items_svd(self, item_user_matrix):
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        item_factors = svd.fit_transform(item_user_matrix)
        self.item_factors = normalize(item_factors)

        sim_dict = {}
        for i in range(self.item_factors.shape[0]):
            sims = self.item_factors @ self.item_factors[i]
            sims[i] = 0
            top_indices = np.argpartition(-sims, self.k)[:self.k]
            sim_dict[self.index_item[i]] = [
                (self.index_item[j], float(sims[j])) for j in top_indices if sims[j] > 0
            ]
        return sim_dict

    def get_similar_items(self, item_id, top_n=10):
        return self.item_sim_matrix.get(item_id, [])[:top_n]

    def recommend_items(self, user_history, top_n=10):
        candidate_scores = {}
        for item in user_history:
            similar_items = self.get_similar_items(item, top_n=self.k)
            for sim_item, score in similar_items:
                if sim_item in user_history:
                    continue
                candidate_scores[sim_item] = candidate_scores.get(sim_item, 0) + score
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ----------------------------
# Item Similarity Function Builder
# ----------------------------
def build_similarity_lookup(sim_matrix):
    def similarity_func(item1, item2):
        if item1 not in sim_matrix:
            return 0.0
        for neighbor, score in sim_matrix[item1]:
            if neighbor == item2:
                return score
        return 0.0
    return similarity_func

# ----------------------------
# Evaluation Metrics
# ----------------------------
def compute_rmse(true_ratings, predicted_ratings):
    return mean_squared_error(true_ratings, predicted_ratings, squared=False)

def precision_recall_at_k(model, test_df, train_df, k=10):
    precisions, recalls = [], []
    for user in test_df['user_id'].unique():
        test_items = test_df[test_df['user_id'] == user]['item_id'].tolist()
        user_history = train_df[train_df['user_id'] == user]['item_id'].unique().tolist()
        recs = [item for item, _ in model.recommend_items(user_history, top_n=k)]
        hits = len(set(recs) & set(test_items))
        precisions.append(hits / k)
        recalls.append(hits / len(test_items) if test_items else 0)
    return np.mean(precisions), np.mean(recalls)

def ndcg_at_k(model, test_df, train_df, k=10):
    ndcg_scores = []
    for user in test_df['user_id'].unique():
        test_items = set(test_df[test_df['user_id'] == user]['item_id'].tolist())
        user_history = train_df[train_df['user_id'] == user]['item_id'].unique().tolist()
        recs = [item for item, _ in model.recommend_items(user_history, top_n=k)]
        dcg = sum((1 / log2(i + 2)) for i, item in enumerate(recs) if item in test_items)
        idcg = sum((1 / log2(i + 2)) for i in range(min(len(test_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def hit_rate_at_k(model, test_df, train_df, k=10):
    hits = []
    for user in test_df['user_id'].unique():
        test_items = test_df[test_df['user_id'] == user]['item_id'].tolist()
        user_history = train_df[train_df['user_id'] == user]['item_id'].unique().tolist()
        recs = [item for item, _ in model.recommend_items(user_history, top_n=k)]
        hit = int(any(item in recs for item in test_items))
        hits.append(hit)
    return np.mean(hits)

def time_to_first_best(model, full_df, test_df, max_steps=5):
    user_times = []
    for user in test_df['user_id'].unique():
        user_data = full_df[full_df['user_id'] == user]
        item_list = user_data['item_id'].tolist()
        if len(item_list) < 2:
            continue
        test_item = item_list[-1]
        history = []
        for i in range(min(len(item_list) - 1, max_steps)):
            history.append(item_list[i])
            recs = [item for item, _ in model.recommend_items(history, top_n=10)]
            if test_item in recs:
                user_times.append(i + 1)
                break
        else:
            user_times.append(max_steps + 1)
    return np.mean(user_times)

def item_coverage(model, all_items, users, train_df, k=10):
    recommended_items = set()
    for user in users:
        user_history = train_df[train_df['user_id'] == user]['item_id'].unique().tolist()
        recs = [item for item, _ in model.recommend_items(user_history, top_n=k)]
        recommended_items.update(recs)
    return len(recommended_items) / len(all_items)

def intra_list_diversity(model, train_df, item_similarity_func, users, k=10):
    diversities = []
    for user in users:
        user_history = train_df[train_df['user_id'] == user]['item_id'].unique().tolist()
        recs = [item for item, _ in model.recommend_items(user_history, top_n=k)]
        if len(recs) <= 1:
            continue
        total_dissim = 0
        count = 0
        for i, j in combinations(recs, 2):
            sim = item_similarity_func(i, j)
            total_dissim += (1 - sim)
            count += 1
        if count > 0:
            diversities.append(total_dissim / count)
    return np.mean(diversities)

# ----------------------------
# Evaluation Wrapper
# ----------------------------
def evaluate_model(model, train_df, test_df, full_df, all_items, users, item_similarity_func, k=10, max_steps=5):
    print("Evaluating model performance...")
    true_ratings, predicted_ratings = [], []
    for _, row in test_df.iterrows():
        user_history = train_df[train_df['user_id'] == row['user_id']]['item_id'].unique().tolist()
        recs = dict(model.recommend_items(user_history, top_n=k))
        pred_rating = recs.get(row['item_id'], 0)
        predicted_ratings.append(pred_rating)
        true_ratings.append(row['rating'])

    rmse = compute_rmse(true_ratings, predicted_ratings)
    precision, recall = precision_recall_at_k(model, test_df, train_df, k=k)
    ndcg = ndcg_at_k(model, test_df, train_df, k=k)
    hitrate = hit_rate_at_k(model, test_df, train_df, k=k)
    time_to_best = time_to_first_best(model, full_df, test_df, max_steps=max_steps)
    coverage = item_coverage(model, all_items, users, train_df, k=k)
    diversity = intra_list_diversity(model, train_df, item_similarity_func, users, k=k)

    return {
        "RMSE": rmse,
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg,
        "HitRate@10": hitrate,
        "TimeToFirstBest": time_to_best,
        "ItemCoverage": coverage,
        "IntraListDiversity": diversity
    }

# ----------------------------
# Sample Usage Entry Point
# ----------------------------
if __name__ == "__main__":
    # Assume create_efficient_dataloader() is defined elsewhere
    train_df, test_df, full_df = create_efficient_dataloader(data_path="/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json", batch_size=50000)

    model = ItemToItemRecommender(k=10)
    model.fit(train_df)

    item_similarity_func = build_similarity_lookup(model.item_sim_matrix)

    results = evaluate_model(
        model=model,
        train_df=train_df,
        test_df=test_df,
        full_df=full_df,
        all_items=set(full_df['item_id'].unique()),
        users=test_df['user_id'].unique().tolist(),
        item_similarity_func=item_similarity_func
    )

    print(results)

