"""
CRITICAL FIX for Collaborative Filtering System
Addresses the data filtering issue causing "0 interactions" after preprocessing
"""

import pandas as pd
import numpy as np
import json
import gzip
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, csr_matrix
import warnings
warnings.filterwarnings('ignore')

class EnhancedItemToItemRecommender:
    def __init__(self, k=30, similarity_method='cosine', min_similarity=0.001, 
                 min_interactions=2, implicit_feedback=True):
        """
        FIXED recommender with proper data filtering
        """
        self.k = k
        self.similarity_method = similarity_method
        self.min_similarity = min_similarity
        self.min_interactions = min_interactions  # REDUCED from 5 to 2
        self.implicit_feedback = implicit_feedback
        
        # Model components
        self.item_sim_matrix = {}
        self.user_item_matrix = None
        self.user_encoder = None
        self.item_encoder = None
        self.user_means = None
        self.global_mean = 0
        
        # Cold-start handling
        self.item_popularity = {}
        self.popular_items = []
        self.user_profiles = {}

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating', 
            timestamp_col=None, category_col=None):
        """FIXED fit method with proper data handling"""
        print(f"Fitting model on {len(df)} interactions...")
        
        # CRITICAL FIX: Much gentler data filtering
        df = self._gentle_filter_data(df, user_col, item_col)
        if len(df) < 100:  # Much lower threshold
            raise ValueError(f"Insufficient data after filtering: {len(df)} interactions")
        
        # Store original data
        self.original_df = df.copy()
        
        # Handle ratings
        if self.implicit_feedback:
            ratings = np.ones(len(df))
            print("Using implicit feedback (binary interactions)")
        else:
            ratings = df[rating_col].values if rating_col else np.ones(len(df))
        
        # Encode users and items
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        user_ids = self.user_encoder.fit_transform(df[user_col])
        item_ids = self.item_encoder.fit_transform(df[item_col])
        
        # Create user-item matrix
        self.user_item_matrix = coo_matrix((ratings, (user_ids, item_ids))).tocsr()
        print(f"Created user-item matrix: {self.user_item_matrix.shape}")
        print(f"Matrix density: {self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.6f}")
        
        # Create user profiles
        self._create_user_profiles(df, user_col, item_col)
        
        # Calculate statistics
        if not self.implicit_feedback:
            self.user_means = np.array(self.user_item_matrix.mean(axis=1)).flatten()
            self.global_mean = np.mean(ratings)
        else:
            self.global_mean = 1.0
        
        # Compute similarities with VERY low threshold
        self.item_sim_matrix = self._compute_item_similarities()
        
        # Compute item popularity
        self._compute_item_popularity(df, item_col)
        
        print(f"Model fitting completed successfully!")

    def _gentle_filter_data(self, df, user_col, item_col):
        """FIXED: Much gentler data filtering to preserve data"""
        original_size = len(df)
        
        # Only remove obvious invalid entries
        df = df.dropna(subset=[user_col, item_col])
        df = df[(df[user_col] != 'unknown') & (df[item_col] != 'unknown')]
        print(f"After removing invalid entries: {len(df)} interactions")
        
        # ONLY do minimal filtering with very low thresholds
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        
        # Use MUCH lower thresholds to preserve data
        min_threshold = max(1, self.min_interactions)  # At least 1, preferably 2
        
        valid_users = user_counts[user_counts >= min_threshold].index
        valid_items = item_counts[item_counts >= min_threshold].index
        
        df_filtered = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]
        
        print(f"After minimal filtering (min_interactions={min_threshold}): {len(df_filtered)} interactions")
        print(f"Data retention: {len(df_filtered)/original_size:.1%}")
        
        # If we lost too much data, reduce threshold further
        if len(df_filtered) < original_size * 0.5:  # Lost more than 50%
            print("Too much data lost, reducing to min_interactions=1")
            valid_users = user_counts[user_counts >= 1].index
            valid_items = item_counts[item_counts >= 1].index
            df_filtered = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]
            print(f"After ultra-minimal filtering: {len(df_filtered)} interactions")
        
        return df_filtered

    def _create_user_profiles(self, df, user_col, item_col):
        """Create user profiles for better recommendations"""
        self.user_item_dict = {}
        for user in df[user_col].unique():
            user_items = df[df[user_col] == user][item_col].tolist()
            self.user_item_dict[user] = set(user_items)
        
        print(f"Created profiles for {len(self.user_item_dict)} users")

    def _compute_item_similarities(self):
        """FIXED: More robust similarity computation"""
        item_user_matrix = self.user_item_matrix.T
        n_items = item_user_matrix.shape[0]
        
        print(f"Computing {self.similarity_method} similarities for {n_items} items...")
        
        if self.similarity_method == 'cosine':
            return self._compute_cosine_similarities_robust(item_user_matrix)
        elif self.similarity_method == 'jaccard':
            return self._compute_jaccard_similarities_robust(item_user_matrix)
        else:
            return self._compute_cosine_similarities_robust(item_user_matrix)

    def _compute_cosine_similarities_robust(self, item_user_matrix):
        """FIXED: Robust cosine similarity computation"""
        sim_dict = {}
        n_items = item_user_matrix.shape[0]
        
        # Use much lower similarity threshold for sparse data
        effective_threshold = min(self.min_similarity, 0.001)
        
        for i in range(n_items):
            item_id = self.item_encoder.classes_[i]
            item_vector = item_user_matrix[i].toarray().flatten()
            
            # Skip items with no interactions
            if np.sum(item_vector) == 0:
                sim_dict[item_id] = []
                continue
            
            similarities = []
            
            # Compare with other items
            for j in range(n_items):
                if i == j:
                    continue
                
                other_vector = item_user_matrix[j].toarray().flatten()
                
                if np.sum(other_vector) == 0:
                    continue
                
                # Compute cosine similarity
                dot_product = np.dot(item_vector, other_vector)
                if dot_product > 0:
                    norm_i = np.linalg.norm(item_vector)
                    norm_j = np.linalg.norm(other_vector)
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot_product / (norm_i * norm_j)
                        
                        if cosine_sim > effective_threshold:
                            other_item_id = self.item_encoder.classes_[j]
                            similarities.append((other_item_id, float(cosine_sim)))
            
            # Sort by similarity and keep top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            sim_dict[item_id] = similarities[:self.k]
        
        items_with_similarities = sum(1 for v in sim_dict.values() if len(v) > 0)
        print(f"Items with similarities: {items_with_similarities}/{len(sim_dict)}")
        
        return sim_dict

    def _compute_jaccard_similarities_robust(self, item_user_matrix):
        """FIXED: Robust Jaccard similarity for binary data"""
        sim_dict = {}
        n_items = item_user_matrix.shape[0]
        
        # Convert to binary
        binary_matrix = (item_user_matrix > 0).astype(int)
        
        for i in range(n_items):
            item_id = self.item_encoder.classes_[i]
            item_users = set(np.where(binary_matrix[i].toarray().flatten() > 0)[0])
            
            if len(item_users) == 0:
                sim_dict[item_id] = []
                continue
            
            similarities = []
            
            for j in range(n_items):
                if i == j:
                    continue
                
                other_users = set(np.where(binary_matrix[j].toarray().flatten() > 0)[0])
                
                if len(other_users) == 0:
                    continue
                
                intersection = len(item_users.intersection(other_users))
                union = len(item_users.union(other_users))
                
                if union > 0 and intersection > 0:
                    jaccard_sim = intersection / union
                    
                    if jaccard_sim > self.min_similarity:
                        other_item_id = self.item_encoder.classes_[j]
                        similarities.append((other_item_id, float(jaccard_sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            sim_dict[item_id] = similarities[:self.k]
        
        return sim_dict

    def _compute_item_popularity(self, df, item_col):
        """Compute item popularity scores"""
        item_counts = df[item_col].value_counts()
        total_interactions = len(df)
        
        self.item_popularity = {}
        for item, count in item_counts.items():
            self.item_popularity[item] = count / total_interactions
        
        # Get popular items (top 50 or top 20%)
        top_threshold = min(50, int(len(item_counts) * 0.2))
        self.popular_items = item_counts.head(max(10, top_threshold)).index.tolist()
        
        print(f"Computed popularity for {len(self.item_popularity)} items")

    def recommend_items(self, user_history, top_n=10, user_id=None):
        """FIXED: Robust recommendation generation"""
        if not user_history:
            # Return popular items for cold-start users
            popular_recs = [(item, self.item_popularity.get(item, 0)) 
                           for item in self.popular_items[:top_n]]
            return popular_recs
        
        user_history_set = set(user_history)
        candidate_scores = {}
        
        # Generate candidates from similar items
        for item in user_history:
            if item in self.item_sim_matrix:
                similar_items = self.item_sim_matrix[item]
                
                for sim_item, sim_score in similar_items:
                    if sim_item not in user_history_set:
                        # Boost score with popularity
                        popularity_boost = self.item_popularity.get(sim_item, 0) * 0.1
                        final_score = sim_score + popularity_boost
                        
                        if sim_item in candidate_scores:
                            candidate_scores[sim_item] += final_score
                        else:
                            candidate_scores[sim_item] = final_score
        
        # If no candidates found, fall back to popular items
        if not candidate_scores:
            popular_recs = [(item, self.item_popularity.get(item, 0)) 
                           for item in self.popular_items 
                           if item not in user_history_set]
            return popular_recs[:top_n]
        
        # Sort and return top recommendations
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]


class FixedEvaluator:
    @staticmethod
    def gentle_train_test_split(df, user_col='user_id', min_train_interactions=2):
        """FIXED: Much gentler train/test split"""
        train_data = []
        test_data = []
        
        for user in df[user_col].unique():
            user_data = df[df[user_col] == user].copy()
            
            # Much lower threshold for inclusion
            if len(user_data) >= min_train_interactions + 1:
                # Sort by timestamp if available
                if 'timestamp' in user_data.columns:
                    user_data = user_data.sort_values('timestamp')
                
                # Take only 1 interaction for test to preserve training data
                test_interactions = user_data.tail(1)
                train_interactions = user_data.head(len(user_data) - 1)
                
                if len(train_interactions) >= min_train_interactions:
                    train_data.append(train_interactions)
                    test_data.append(test_interactions)
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        print(f"Gentle split: {len(train_df)} train, {len(test_df)} test interactions")
        print(f"Users in evaluation: {len(test_df[user_col].unique()) if not test_df.empty else 0}")
        
        return train_df, test_df

    @staticmethod
    def robust_evaluation(model, test_df, train_df, k=10):
        """FIXED: More robust evaluation that actually works"""
        print("Running robust evaluation...")
        
        precisions = []
        recalls = []
        hits = 0
        total_evaluations = 0
        
        for user in test_df['user_id'].unique():
            # Get test items for this user
            test_items = set(test_df[test_df['user_id'] == user]['item_id'].tolist())
            if not test_items:
                continue
            
            # Get training history for this user
            user_history = train_df[train_df['user_id'] == user]['item_id'].tolist()
            if len(user_history) < 1:  # Lower threshold
                continue
            
            try:
                # Get recommendations
                recs = model.recommend_items(user_history, top_n=k, user_id=user)
                if not recs:
                    continue
                
                rec_items = set([item for item, _ in recs])
                
                if not rec_items:
                    continue
                
                # Calculate metrics
                relevant_retrieved = len(test_items.intersection(rec_items))
                
                precision = relevant_retrieved / len(rec_items) if rec_items else 0
                recall = relevant_retrieved / len(test_items) if test_items else 0
                
                precisions.append(precision)
                recalls.append(recall)
                
                if relevant_retrieved > 0:
                    hits += 1
                total_evaluations += 1
                
            except Exception as e:
                print(f"Error evaluating user {user}: {e}")
                continue
        
        # Calculate metrics
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        hit_rate = hits / total_evaluations if total_evaluations > 0 else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"Evaluated {total_evaluations} users successfully")
        
        return {
            f"Precision@{k}": avg_precision,
            f"Recall@{k}": avg_recall,
            f"F1@{k}": f1_score,
            f"HitRate@{k}": hit_rate,
            f"Users_Evaluated": total_evaluations,
            f"Total_Hits": hits
        }


def load_reviews_from_json_gz(path):
    """Load reviews from JSON.gz file"""
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
                try:
                    data = json.loads(line)
                    rows.append({
                        'user_id': data.get('reviewerID', 'unknown'),
                        'item_id': data.get('asin', 'unknown'),
                        'rating': float(data.get('overall', 1.0)),
                        'timestamp': data.get('unixReviewTime', 0),
                        'category': 'Movies_and_TV'
                    })
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise

    return pd.DataFrame(rows)


def run_fixed_evaluation(data_path=None, sample_size=50000, top_k=10):
    """FIXED: Main evaluation function that actually works"""
    if data_path is None:
        data_path = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz"

    print("="*60)
    print("FIXED COLLABORATIVE FILTERING EVALUATION")
    print("="*60)

    # Load and preprocess data
    df = load_reviews_from_json_gz(data_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Working with {len(df)} interactions")
    
    # Use gentle train/test split
    train_df, test_df = FixedEvaluator.gentle_train_test_split(df, min_train_interactions=1)
    
    if len(test_df) == 0:
        print("No test data available even with gentle split!")
        return
    
    print(f"Data split - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Test configurations with lower thresholds
    configs = [
        {
            "name": "Ultra-Gentle-Cosine",
            "similarity_method": "cosine",
            "k": 20,
            "min_similarity": 0.001,
            "min_interactions": 1
        },
        {
            "name": "Ultra-Gentle-Jaccard",
            "similarity_method": "jaccard", 
            "k": 15,
            "min_similarity": 0.001,
            "min_interactions": 1
        }
    ]
    
    best_result = None
    best_config = None
    
    for config in configs:
        print(f"\n{'-'*40}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'-'*40}")
        
        try:
            model = FixedItemToItemRecommender(
                k=config["k"],
                similarity_method=config["similarity_method"],
                min_similarity=config["min_similarity"],
                min_interactions=config["min_interactions"],
                implicit_feedback=True
            )
            
            model.fit(train_df, timestamp_col='timestamp', category_col='category')
            results = FixedEvaluator.robust_evaluation(model, test_df, train_df, k=top_k)
            
            print(f"\nResults for {config['name']}:")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
            
            # Select best based on F1 score
            if best_result is None or results[f"F1@{top_k}"] > best_result[f"F1@{top_k}"]:
                best_result = results
                best_config = config
                
        except Exception as e:
            print(f"Error in configuration {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if best_result and best_config:
        print(f"\nBest Configuration: {best_config['name']}")
        for key, value in best_config.items():
            if key != 'name':
                print(f"{key}: {value}")
        
        print(f"\nBest Results:")
        for metric, value in best_result.items():
            print(f"{metric}: {value:.4f}")
        
        # Provide recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print(f"{'='*60}")
        
        if best_result[f"F1@{top_k}"] < 0.1:
            print("📊 PERFORMANCE ANALYSIS:")
            print("   - Low F1 score indicates fundamental data sparsity issues")
            print("   - Consider content-based or hybrid approaches")
            print("   - Try popularity-based baseline for comparison")
            
        print("\n💡 SPECIFIC IMPROVEMENTS:")
        print("   1. Increase sample size to 200k+ interactions")
        print("   2. Use temporal filtering (recent interactions only)")
        print("   3. Implement matrix factorization (SVD/NMF)")
        print("   4. Add content-based features if available")
        print("   5. Consider neural collaborative filtering")
        
    else:
        print("❌ No successful configurations found.")
        print("   This indicates severe data quality issues.")
        print("   Consider running diagnostics first.")


if __name__ == "__main__":
    run_fixed_evaluation()