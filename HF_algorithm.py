from __future__ import annotations
import gzip
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def enhanced_preprocess_text(text: str, summary: str = "") -> str:
    """Ultra-enhanced text preprocessing with sentiment and semantic features"""
    if not text and not summary:
        return ""
    
    # Combine text and summary with higher weight on summary
    combined = f"{text} {summary} {summary} {summary}"
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(combined)
    
    # Add sentiment keywords based on score
    if sentiment['compound'] >= 0.8:
        combined += " " + "excellent amazing fantastic incredible perfect wonderful outstanding superb" * 3
    elif sentiment['compound'] >= 0.5:
        combined += " " + "great good nice enjoyable pleasant satisfying" * 2
    elif sentiment['compound'] <= -0.5:
        combined += " " + "terrible awful horrible disappointing poor bad" * 2
    
    # Convert to lowercase
    combined = combined.lower()
    
    # Remove URLs
    combined = re.sub(r'http\S+', '', combined)
    
    # Keep alphanumeric and important punctuation
    combined = re.sub(r"[^a-z0-9\s.!?'-]", " ", combined)
    
    # Tokenize and process
    tokens = combined.split()
    
    # Important keywords to preserve
    important_words = {
        'not', 'no', 'very', 'really', 'too', 'so', 'but', 'however',
        'excellent', 'amazing', 'terrible', 'awful', 'best', 'worst',
        'love', 'hate', 'perfect', 'horrible', 'fantastic', 'disappointing'
    }
    
    # Keep important words and remove common stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words or word in important_words]
    
    # Add bigrams for important phrases
    bigrams = []
    for i in range(len(tokens)-1):
        if tokens[i] in ['very', 'really', 'extremely', 'not']:
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
    
    return " ".join(tokens + bigrams)

def parse_json_gz(path: str | Path):
    with gzip.open(path, "rb") as f:
        for line in f:
            yield json.loads(line)

def load_dataframe(path: str | Path, sample_users: int | None = None) -> pd.DataFrame:
    rows = []
    users_seen: set[str] = set()
    
    for record in parse_json_gz(path):
        if sample_users is not None and len(users_seen) >= sample_users:
            if record["reviewerID"] not in users_seen:
                continue
        users_seen.add(record["reviewerID"])
        
        # Extract all available features
        review_text = record.get("reviewText", "")
        summary = record.get("summary", "")
        
        rows.append({
            "user": record["reviewerID"],
            "item": record["asin"],
            "rating": record.get("overall", 0.0),
            "text": enhanced_preprocess_text(review_text, summary),
            "timestamp": record.get("unixReviewTime", 0),
            "helpful": record.get("helpful", [0, 0]),
            "verified": record.get("verified", False),
        })

    df = pd.DataFrame(rows)
    df.dropna(subset=["user", "item"], inplace=True)
    df.sort_values(["user", "timestamp"], inplace=True)
    
    # Calculate helpfulness ratio
    df["helpfulness"] = df["helpful"].apply(
        lambda x: x[0] / x[1] if x[1] > 0 else 0.5
    )
    
    # Calculate user expertise based on helpfulness
    user_expertise = df.groupby("user")["helpfulness"].mean()
    df["user_expertise"] = df["user"].map(user_expertise)

    # Aggressive filtering for quality
    user_counts = df["user"].value_counts()
    item_counts = df["item"].value_counts()
    
    # Keep only users and items with sufficient interactions
    min_user_interactions = 10  # Increased significantly
    min_item_interactions = 10
    
    df = df[df["user"].isin(user_counts[user_counts >= min_user_interactions].index)]
    df = df[df["item"].isin(item_counts[item_counts >= min_item_interactions].index)]
    
    print(f"After filtering: {len(df)} interactions, {df['user'].nunique()} users, {df['item'].nunique()} items")

    return df

def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced temporal split that ensures test items exist in train"""
    grouped = df.groupby("user")
    train_list, test_list = [], []
    
    for user, group in grouped:
        if len(group) < 5:  # Need enough interactions
            continue
        
        # Sort by timestamp
        group = group.sort_values("timestamp")
        
        # Take last 20% for test, but ensure we have enough training data
        test_size = max(1, min(int(len(group) * test_ratio), len(group) - 3))
        
        test_items = group.tail(test_size)
        train_items = group.head(len(group) - test_size)
        
        # Ensure test items exist in training data (for other users)
        test_list.append(test_items)
        train_list.append(train_items)
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    # Remove test entries for items that don't exist in train
    train_items = set(train_df["item"].unique())
    test_df = test_df[test_df["item"].isin(train_items)]
    
    return train_df, test_df

def build_enhanced_user_item_matrix(train: pd.DataFrame):
    """Build highly optimized user-item matrix with advanced weighting"""
    users = {u: i for i, u in enumerate(sorted(train["user"].unique()))}
    items = {i: j for j, i in enumerate(sorted(train["item"].unique()))}
    
    row = train["user"].map(users).values
    col = train["item"].map(items).values
    
    # Advanced weighting scheme
    ratings = train["rating"].values
    timestamps = train["timestamp"].values
    helpfulness = train["helpfulness"].values
    verified = train["verified"].astype(float).values
    expertise = train["user_expertise"].values
    
    # Normalize timestamps
    time_weights = 1 + 0.5 * (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1)
    
    # Combine multiple signals
    data = (
        (ratings / 5.0) *  # Normalized ratings
        np.sqrt(time_weights) *  # Recency boost
        (1 + 0.3 * helpfulness) *  # Helpfulness boost
        (1 + 0.2 * verified) *  # Verified purchase boost
        (1 + 0.2 * expertise)  # User expertise boost
    )
    
    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    
    # Normalize by user activity
    user_norms = np.array(matrix.sum(axis=1)).flatten()
    user_norms[user_norms == 0] = 1
    
    # Create diagonal matrix for normalization
    norm_matrix = csr_matrix((1 / user_norms, (range(len(users)), range(len(users)))), 
                            shape=(len(users), len(users)))
    matrix = norm_matrix @ matrix
    
    # Ensure it's CSR format
    matrix = matrix.tocsr()
    
    return matrix, users, items

def build_advanced_item_features(train: pd.DataFrame, items: Dict[str, int]):
    """Build ultra-advanced item features with multiple modalities"""
    # Aggregate all information by item
    item_aggs = train.groupby("item").agg({
        "text": lambda x: " ".join(x),
        "rating": ["mean", "std", "count", "min", "max"],
        "helpfulness": ["mean", "std"],
        "verified": "mean",
        "user_expertise": "mean"
    })
    
    # Flatten column names
    item_aggs.columns = ['_'.join(col).strip() for col in item_aggs.columns.values]
    
    # Create rich text features for each item
    text_features = []
    
    for item in items.keys():
        if item in item_aggs.index:
            text = item_aggs.loc[item, "text_<lambda>"]
            
            # Add rating-based keywords
            avg_rating = item_aggs.loc[item, "rating_mean"]
            rating_std = item_aggs.loc[item, "rating_std"]
            
            # High-quality items
            if avg_rating >= 4.5 and rating_std < 0.5:
                text += " exceptional outstanding excellent perfect amazing " * 5
            elif avg_rating >= 4.0:
                text += " great good quality recommended " * 3
            elif avg_rating <= 2.0:
                text += " poor bad terrible awful disappointing " * 3
            
            # Add popularity signals
            count = item_aggs.loc[item, "rating_count"]
            if count > train.groupby("item").size().quantile(0.9):
                text += " popular bestseller trending " * 2
            
            # Add consistency signals
            if rating_std < 0.5:
                text += " consistent reliable dependable " * 2
                
            text_features.append(text)
        else:
            text_features.append("")
    
    # Ultra-enhanced TF-IDF
    tfidf = TfidfVectorizer(
        stop_words=None,  # We already processed stopwords
        max_features=50000,  # Much larger vocabulary
        ngram_range=(1, 4),  # Up to 4-grams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True
    )
    
    tfidf_matrix = tfidf.fit_transform(text_features)
    
    # Create comprehensive metadata features
    metadata_features = []
    
    for item in items.keys():
        if item in item_aggs.index:
            features = [
                # Rating features
                item_aggs.loc[item, "rating_mean"] / 5.0,
                1 - (item_aggs.loc[item, "rating_std"] / 2.5) if not pd.isna(item_aggs.loc[item, "rating_std"]) else 0.5,
                min(item_aggs.loc[item, "rating_count"] / 100, 1.0),
                item_aggs.loc[item, "rating_min"] / 5.0,
                item_aggs.loc[item, "rating_max"] / 5.0,
                
                # Quality indicators
                item_aggs.loc[item, "helpfulness_mean"],
                1 - (item_aggs.loc[item, "helpfulness_std"] if not pd.isna(item_aggs.loc[item, "helpfulness_std"]) else 0.5),
                item_aggs.loc[item, "verified_mean"],
                item_aggs.loc[item, "user_expertise_mean"],
                
                # Derived features
                (item_aggs.loc[item, "rating_mean"] - 3) / 2,  # Centered rating
                int(item_aggs.loc[item, "rating_mean"] >= 4),  # High quality binary
                int(item_aggs.loc[item, "rating_count"] > 20),  # Popular binary
            ]
        else:
            features = [0.5] * 12
            
        metadata_features.append(features)
    
    # Scale metadata features
    scaler = MinMaxScaler()
    metadata_scaled = scaler.fit_transform(metadata_features)
    metadata_matrix = csr_matrix(metadata_scaled)
    
    # Combine features with optimal weights
    from scipy.sparse import hstack
    combined_features = hstack([
        tfidf_matrix * 0.7,  # Text features
        metadata_matrix * 0.3  # Metadata features
    ])
    
    return combined_features, tfidf, item_aggs

def advanced_matrix_factorization(user_item_matrix: csr_matrix, n_factors: int = 100):
    """Multiple matrix factorization techniques combined"""
    # SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors_svd = svd.fit_transform(user_item_matrix)
    item_factors_svd = svd.components_.T
    
    # NMF for non-negative factors
    nmf = NMF(n_components=n_factors // 2, random_state=42, max_iter=300)
    user_factors_nmf = nmf.fit_transform(user_item_matrix)
    item_factors_nmf = nmf.components_.T
    
    # Combine factors
    user_factors = np.hstack([user_factors_svd, user_factors_nmf])
    item_factors = np.vstack([item_factors_svd.T, item_factors_nmf.T]).T
    
    return user_factors, item_factors

def compute_item_item_similarity(item_features: csr_matrix, n_neighbors: int = 50):
    """Precompute item-item similarities for efficiency"""
    # Use approximate nearest neighbors for speed
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    nn_model.fit(item_features)
    
    return nn_model

def ultra_hybrid_recommend(
    user: str,
    k: int,
    user_item_matrix: csr_matrix,
    user_index: Dict[str, int],
    item_index: Dict[str, int],
    item_features: csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    item_nn_model: NearestNeighbors,
    train_df: pd.DataFrame,
    item_aggs: pd.DataFrame,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.4
) -> List[str]:
    """Ultra-advanced hybrid recommendation system"""
    
    inv_item_index = {j: i for i, j in item_index.items()}
    n_items = len(item_index)
    
    # Handle cold start
    if user not in user_index:
        # Return best items based on rating and popularity
        best_items = item_aggs.nlargest(k * 2, "rating_mean")
        best_items = best_items[best_items["rating_count"] >= 10]
        return list(best_items.head(k).index)
    
    u_idx = user_index[user]
    user_vec = user_item_matrix[u_idx]
    seen_items = set(user_vec.indices)
    
    # Initialize score components
    scores = np.zeros(n_items)
    
    # 1. Enhanced Collaborative Filtering
    if user_vec.nnz > 0:
        # Find very similar users
        nn = NearestNeighbors(n_neighbors=min(200, user_item_matrix.shape[0] // 5), 
                             metric='cosine', algorithm='brute')
        nn.fit(user_item_matrix)
        distances, neighbors = nn.kneighbors(user_vec)
        
        # Use only very similar users (distance < 0.5)
        similar_mask = distances[0] < 0.5
        similar_users = neighbors[0][similar_mask]
        similar_distances = distances[0][similar_mask]
        
        if len(similar_users) > 1:
            similarities = 1 - similar_distances
            # Exponential weighting for very similar users
            similarities = np.exp(2 * similarities) - 1
            
            cf_scores = np.zeros(n_items)
            for sim, neighbor in zip(similarities[1:], similar_users[1:]):
                neighbor_vec = user_item_matrix[neighbor].toarray().flatten()
                cf_scores += sim * neighbor_vec
            
            cf_scores = cf_scores / similarities[1:].sum()
            scores += alpha * cf_scores
    
    # 2. Advanced Content-Based Filtering
    if len(seen_items) > 0:
        seen_indices = list(seen_items)
        user_items = train_df[train_df["user"] == user]
        
        # Build sophisticated user profile
        # Weight by rating and recency
        ratings_dict = dict(zip(user_items["item"], user_items["rating"]))
        timestamps_dict = dict(zip(user_items["item"], user_items["timestamp"]))
        
        weights = []
        for idx in seen_indices:
            item_id = inv_item_index.get(idx)
            if item_id and item_id in ratings_dict:
                rating_weight = ratings_dict[item_id] / 5.0
                # Exponential weight for high ratings
                if ratings_dict[item_id] >= 4:
                    rating_weight = rating_weight ** 2
                
                # Recency weight
                if item_id in timestamps_dict:
                    time_weight = 1.0  # Can add time decay if needed
                else:
                    time_weight = 0.8
                    
                weights.append(rating_weight * time_weight)
            else:
                weights.append(0.1)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        # Create user profile
        user_profile = item_features[seen_indices].T @ weights
        user_profile = user_profile.T.reshape(1, -1)
        
        # Content similarity
        cb_scores = cosine_similarity(user_profile, item_features).flatten()
        
        # Boost items similar to loved items (rating >= 4.5)
        loved_items = user_items[user_items["rating"] >= 4.5]["item"].values
        if len(loved_items) > 0:
            loved_indices = [item_index[i] for i in loved_items if i in item_index]
            if loved_indices:
                love_boost = cosine_similarity(item_features[loved_indices], item_features).max(axis=0)
                cb_scores += 0.3 * love_boost
        
        scores += beta * cb_scores
    
    # 3. Matrix Factorization Score
    mf_scores = user_factors[u_idx] @ item_factors.T
    mf_scores = (mf_scores - mf_scores.min()) / (mf_scores.max() - mf_scores.min() + 1e-8)
    scores += gamma * mf_scores
    
    # 4. Quality Boost
    quality_boost = np.zeros(n_items)
    for idx, item_id in inv_item_index.items():
        if item_id in item_aggs.index:
            # Strong quality signal
            rating = item_aggs.loc[item_id, "rating_mean"]
            count = item_aggs.loc[item_id, "rating_count"]
            std = item_aggs.loc[item_id, "rating_std"]
            
            if rating >= 4.0 and count >= 10 and std < 1.0:
                quality_boost[idx] = (rating - 3.5) * np.log1p(count) / 10
    
    scores += 0.2 * quality_boost
    
    # 5. Exploration bonus for diverse items
    if k > 5:
        item_popularity = train_df["item"].value_counts()
        exploration_bonus = np.zeros(n_items)
        
        for idx, item_id in inv_item_index.items():
            if item_id in item_popularity:
                # Inverse popularity for exploration
                pop_rank = item_popularity.rank(pct=True)[item_id]
                if 0.3 < pop_rank < 0.7:  # Mid-popularity items
                    exploration_bonus[idx] = 0.1
        
        scores += exploration_bonus
    
    # 6. Remove seen items
    scores[list(seen_items)] = -np.inf
    
    # 7. Diversified selection
    if k <= 5:
        # For small k, just take top scores
        top_items = np.argpartition(-scores, k)[:k]
        top_items = top_items[np.argsort(-scores[top_items])]
    else:
        # MMR-based diverse selection
        selected = []
        candidates = list(range(n_items))
        
        # Remove seen items from candidates
        candidates = [c for c in candidates if c not in seen_items]
        
        # First item: highest score
        if candidates:
            best_idx = max(candidates, key=lambda x: scores[x])
            selected.append(best_idx)
            candidates.remove(best_idx)
        
        # Subsequent items: balance score and diversity
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for c in candidates:
                score = scores[c]
                
                # Diversity penalty
                if selected:
                    similarities = cosine_similarity(
                        item_features[c:c+1],
                        item_features[selected]
                    ).flatten()
                    max_sim = similarities.max()
                    diversity = 1 - max_sim
                else:
                    diversity = 1.0
                
                # MMR score
                mmr = 0.7 * score + 0.3 * diversity
                mmr_scores.append(mmr)
            
            best_idx = candidates[np.argmax(mmr_scores)]
            selected.append(best_idx)
            candidates.remove(best_idx)
        
        top_items = selected
    
    # Convert to item IDs
    recommendations = [inv_item_index[i] for i in top_items if i in inv_item_index]
    
    return recommendations[:k]

def evaluate_ultra(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    k: int = 10
) -> pd.DataFrame:
    """Comprehensive evaluation with all requested metrics"""
    print("Building advanced recommendation components...")
    
    # Build all components
    ui_matrix, user_index, item_index = build_enhanced_user_item_matrix(train)
    print("User-item matrix built")
    
    item_features, _, item_aggs = build_advanced_item_features(train, item_index)
    print("Item features extracted")
    
    user_factors, item_factors = advanced_matrix_factorization(ui_matrix, n_factors=100)
    print("Matrix factorization completed")
    
    item_nn_model = compute_item_item_similarity(item_features, n_neighbors=50)
    print("Item similarities computed")
    
    # Calculate item popularity for novelty
    item_popularity = train["item"].value_counts()
    total_interactions = len(train)
    
    # Identify cold-start users and items
    user_interaction_counts = train["user"].value_counts()
    item_interaction_counts = train["item"].value_counts()
    
    cold_start_threshold = 5  # Users/items with <= 5 interactions
    cold_users = set(user_interaction_counts[user_interaction_counts <= cold_start_threshold].index)
    cold_items = set(item_interaction_counts[item_interaction_counts <= cold_start_threshold].index)
    
    # Optimal parameters for high precision
    best_params = [
        {"alpha": 0.3, "beta": 0.3, "gamma": 0.4},
        {"alpha": 0.25, "beta": 0.35, "gamma": 0.4},
        {"alpha": 0.35, "beta": 0.25, "gamma": 0.4},
    ]
    
    results = []
    all_items = set(item_index.keys())
    
    for params in best_params:
        print(f"\nEvaluating with params: {params}")
        
        # Initialize metrics
        metrics = {
            # Accuracy metrics
            "hits": 0, "total_precision": 0, "total_recall": 0, 
            "total_ndcg": 0, "total_map": 0, "total_mrr": 0,
            
            # Cold-start metrics
            "cold_user_hits": 0, "cold_user_precision": 0, "cold_user_count": 0,
            "cold_item_hits": 0, "cold_item_precision": 0, "cold_item_count": 0,
            "time_to_first_good": [],
            
            # Diversity & Coverage
            "all_recommended_items": set(),
            "total_diversity": 0,
            "total_novelty": 0,
            
            # User counts
            "total_users": 0
        }
        
        # Group test by user
        test_grouped = test.groupby("user")
        
        for user, user_test in test_grouped:
            true_items = set(user_test["item"].values)
            
            # Get recommendations
            recs = ultra_hybrid_recommend(
                user, k, ui_matrix, user_index, item_index,
                item_features, user_factors, item_factors,
                item_nn_model, train, item_aggs,
                **params
            )
            
            if not recs:
                continue
            
            metrics["total_users"] += 1
            metrics["all_recommended_items"].update(recs)
            
            # Check if cold-start user
            is_cold_user = user in cold_users
            
            # Calculate hits
            rec_set = set(recs)
            hits_set = rec_set.intersection(true_items)
            n_hits = len(hits_set)
            
            # Cold-start item analysis
            cold_item_hits = len([item for item in hits_set if item in cold_items])
            
            # 1. ACCURACY METRICS
            precision = n_hits / k
            recall = n_hits / len(true_items) if true_items else 0
            
            metrics["total_precision"] += precision
            metrics["total_recall"] += recall
            
            if n_hits > 0:
                metrics["hits"] += 1
                
                # NDCG calculation
                dcg = 0
                for i, item in enumerate(recs):
                    if item in true_items:
                        dcg += 1 / math.log2(i + 2)
                
                # Ideal DCG
                idcg = sum(1 / math.log2(i + 2) for i in range(min(n_hits, k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics["total_ndcg"] += ndcg
                
                # MRR - position of first hit
                first_hit_pos = next(i for i, item in enumerate(recs) if item in true_items)
                metrics["total_mrr"] += 1 / (first_hit_pos + 1)
                
                # MAP - Average Precision
                ap = 0
                n_hits_so_far = 0
                for i, item in enumerate(recs):
                    if item in true_items:
                        n_hits_so_far += 1
                        ap += n_hits_so_far / (i + 1)
                ap = ap / n_hits if n_hits > 0 else 0
                metrics["total_map"] += ap
                
                # Time to first good recommendation
                if is_cold_user:
                    # Count user's previous interactions
                    user_history_length = len(train[train["user"] == user])
                    metrics["time_to_first_good"].append(user_history_length)
            
            # 2. COLD-START METRICS
            if is_cold_user:
                metrics["cold_user_count"] += 1
                if n_hits > 0:
                    metrics["cold_user_hits"] += 1
                    metrics["cold_user_precision"] += precision
            
            if cold_item_hits > 0:
                metrics["cold_item_count"] += 1
                metrics["cold_item_hits"] += cold_item_hits
                metrics["cold_item_precision"] += cold_item_hits / k
            
            # 3. DIVERSITY & NOVELTY METRICS
            if len(recs) > 1:
                # Diversity - average pairwise dissimilarity
                rec_indices = [item_index[item] for item in recs if item in item_index]
                if len(rec_indices) > 1:
                    similarities = cosine_similarity(
                        item_features[rec_indices], 
                        item_features[rec_indices]
                    )
                    # Average of off-diagonal elements
                    mask = np.ones_like(similarities) - np.eye(len(similarities))
                    avg_similarity = (similarities * mask).sum() / mask.sum()
                    diversity = 1 - avg_similarity
                    metrics["total_diversity"] += diversity
            
            # Novelty - inverse popularity
            novelty_scores = []
            for item in recs:
                if item in item_popularity:
                    # Use negative log of popularity
                    novelty = -math.log2(item_popularity[item] / total_interactions)
                else:
                    # New item, maximum novelty
                    novelty = -math.log2(1 / total_interactions)
                novelty_scores.append(novelty)
            
            metrics["total_novelty"] += np.mean(novelty_scores)
        
        if metrics["total_users"] == 0:
            continue
        
        # Calculate final metrics
        n_users = metrics["total_users"]
        
        # Coverage
        coverage = len(metrics["all_recommended_items"]) / len(all_items)
        
        # Average time to first good recommendation
        avg_time_to_good = (np.mean(metrics["time_to_first_good"]) 
                           if metrics["time_to_first_good"] else 0)
        
        results.append({
            # Parameters
            "Alpha": params["alpha"],
            "Beta": params["beta"], 
            "Gamma": params["gamma"],
            
            # 1. ACCURACY METRICS
            "Precision@K": metrics["total_precision"] / n_users,
            "Recall@K": metrics["total_recall"] / n_users,
            "NDCG@K": metrics["total_ndcg"] / n_users,
            "MAP@K": metrics["total_map"] / n_users,
            "MRR@K": metrics["total_mrr"] / n_users,
            "HitRate@K": metrics["hits"] / n_users,
            "F1@K": 2 * (metrics["total_precision"] / n_users) * (metrics["total_recall"] / n_users) / 
                    ((metrics["total_precision"] / n_users) + (metrics["total_recall"] / n_users) + 1e-8),
            
            # 2. COLD-START METRICS
            "Cold_User_Precision@K": (metrics["cold_user_precision"] / metrics["cold_user_count"] 
                                     if metrics["cold_user_count"] > 0 else 0),
            "Cold_User_HitRate": (metrics["cold_user_hits"] / metrics["cold_user_count"]
                                 if metrics["cold_user_count"] > 0 else 0),
            "Cold_Item_Precision@K": (metrics["cold_item_precision"] / metrics["cold_item_count"]
                                     if metrics["cold_item_count"] > 0 else 0),
            "Time_to_First_Good": avg_time_to_good,
            
            # 3. DIVERSITY & COVERAGE METRICS
            "Coverage@K": coverage,
            "Diversity@K": metrics["total_diversity"] / n_users,
            "Novelty@K": metrics["total_novelty"] / n_users,
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    path_to_data = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz"
    sample_users = 5000
    top_k = 10
    
    print("Loading data with ultra-enhanced preprocessing...")
    df = load_dataframe(path_to_data, sample_users=sample_users)
    
    # Use temporal split
    train, test = temporal_split(df, test_ratio=0.2)
    print(f"\nFinal dataset stats:")
    print(f"Train interactions: {len(train):,} | Test interactions: {len(test):,}")
    print(f"Unique users: {train['user'].nunique():,} | Unique items: {train['item'].nunique():,}")
    print(f"Avg ratings per user: {len(train) / train['user'].nunique():.1f}")
    print(f"Avg ratings per item: {len(train) / train['item'].nunique():.1f}")
    
    # Analyze cold-start statistics
    user_counts = train["user"].value_counts()
    item_counts = train["item"].value_counts()
    cold_users = user_counts[user_counts <= 5]
    cold_items = item_counts[item_counts <= 5]
    
    print(f"\nCold-start analysis:")
    print(f"Cold-start users (≤5 interactions): {len(cold_users)} ({len(cold_users)/len(user_counts)*100:.1f}%)")
    print(f"Cold-start items (≤5 interactions): {len(cold_items)} ({len(cold_items)/len(item_counts)*100:.1f}%)")
    
    print("\nEvaluating ultra-enhanced hybrid system with comprehensive metrics...")
    metrics_df = evaluate_ultra(train, test, k=top_k)
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*100)
    
    # Display results grouped by metric type
    for idx, row in metrics_df.iterrows():
        print(f"\n--- Configuration {idx+1}: α={row['Alpha']}, β={row['Beta']}, γ={row['Gamma']} ---")
        
        print("\n📊 ACCURACY METRICS:")
        print(f"  • Precision@{top_k}: {row['Precision@K']:.4f} ({row['Precision@K']*100:.2f}%)")
        print(f"  • Recall@{top_k}: {row['Recall@K']:.4f} ({row['Recall@K']*100:.2f}%)")
        print(f"  • NDCG@{top_k}: {row['NDCG@K']:.4f}")
        print(f"  • MAP@{top_k}: {row['MAP@K']:.4f}")
        print(f"  • MRR@{top_k}: {row['MRR@K']:.4f}")
        print(f"  • Hit Rate@{top_k}: {row['HitRate@K']:.4f} ({row['HitRate@K']*100:.2f}%)")
        print(f"  • F1 Score@{top_k}: {row['F1@K']:.4f}")
        
        print("\n❄️  COLD-START METRICS:")
        print(f"  • Cold User Precision@{top_k}: {row['Cold_User_Precision@K']:.4f}")
        print(f"  • Cold User Hit Rate: {row['Cold_User_HitRate']:.4f} ({row['Cold_User_HitRate']*100:.2f}%)")
        print(f"  • Cold Item Precision@{top_k}: {row['Cold_Item_Precision@K']:.4f}")
        print(f"  • Avg Time to First Good Rec: {row['Time_to_First_Good']:.1f} interactions")
        
        print("\n🌈 DIVERSITY & COVERAGE METRICS:")
        print(f"  • Coverage@{top_k}: {row['Coverage@K']:.4f} ({row['Coverage@K']*100:.2f}% of catalog)")
        print(f"  • Diversity@{top_k}: {row['Diversity@K']:.4f}")
        print(f"  • Novelty@{top_k}: {row['Novelty@K']:.4f}")
    
    # Find best configurations
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS")
    print("="*100)
    
    best_precision = metrics_df.loc[metrics_df["Precision@K"].idxmax()]
    best_ndcg = metrics_df.loc[metrics_df["NDCG@K"].idxmax()]
    best_coverage = metrics_df.loc[metrics_df["Coverage@K"].idxmax()]
    
    print(f"\n🏆 Best for Precision: α={best_precision['Alpha']}, β={best_precision['Beta']}, γ={best_precision['Gamma']}")
    print(f"   → Precision@{top_k}: {best_precision['Precision@K']:.4f}")
    
    print(f"\n🏆 Best for NDCG: α={best_ndcg['Alpha']}, β={best_ndcg['Beta']}, γ={best_ndcg['Gamma']}")
    print(f"   → NDCG@{top_k}: {best_ndcg['NDCG@K']:.4f}")
    
    print(f"\n🏆 Best for Coverage: α={best_coverage['Alpha']}, β={best_coverage['Beta']}, γ={best_coverage['Gamma']}")
    print(f"   → Coverage@{top_k}: {best_coverage['Coverage@K']:.4f}")
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))