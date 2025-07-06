import pandas as pd
import numpy as np
from CF_Module import EnhancedItemToItemRecommender, EnhancedEvaluator
from Dataloader import create_efficient_dataloader
import gzip
import json
from tqdm import tqdm
import glob
import os
import multiprocessing as mp
from collections import defaultdict
import pickle

# Configuration constants
DEFAULT_SAMPLE_SIZE = 200000
DEFAULT_TOP_K = 10
DEFAULT_RANDOM_STATE = 42
MIN_ROWS_THRESHOLD = 1000
MAX_PREPROCESSING_ITERATIONS = 3

def load_reviews_from_json_gz(path):
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
                except json.JSONDecodeError as e:
                    continue
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise

    return pd.DataFrame(rows)

def enhanced_preprocess_data(df, min_user_interactions=1, min_item_interactions=1,
                           max_iterations=MAX_PREPROCESSING_ITERATIONS,
                           preserve_percentage=0.8):
    """Enhanced preprocessing that preserves more data"""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    print(f"Initial shape: {df.shape}")
    
    # Remove only truly invalid data
    df = df[(df['user_id'] != 'unknown') & (df['item_id'] != 'unknown')]
    df = df.dropna(subset=['user_id', 'item_id'])
    print(f"After removing invalid entries: {df.shape}")

    if df.empty:
        raise ValueError("No valid data after removing unknown users/items")

    # Adaptive filtering based on data distribution
    original_size = len(df)
    target_size = int(original_size * preserve_percentage)
    
    for iteration in range(max_iterations):
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        # Dynamic thresholds based on percentiles
        user_threshold = max(min_user_interactions, 
                           int(np.percentile(user_counts, 5)))
        item_threshold = max(min_item_interactions, 
                           int(np.percentile(item_counts, 5)))
        
        prev_size = len(df)
        
        # Filter users and items
        valid_users = user_counts[user_counts >= user_threshold].index
        valid_items = item_counts[item_counts >= item_threshold].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        print(f"Iteration {iteration+1}: {df.shape} (user_thresh={user_threshold}, item_thresh={item_threshold})")
        
        # Stop if we've removed too much data or converged
        if len(df) < target_size or len(df) == prev_size:
            break
    
    print(f"Final preprocessing: {original_size} -> {len(df)} ({len(df)/original_size:.1%} retained)")
    return df

def run_optimized_evaluation(data_path=None, sample_size=DEFAULT_SAMPLE_SIZE, k_values=[5, 10, 20]):
    """Optimized evaluation with best practices for 5-10% performance"""
    if data_path is None:
        data_path = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz"

    print("="*60)
    print("OPTIMIZED CF EVALUATION - TARGET: 5-10% METRICS")
    print("="*60)
    
    # Load data
    df = load_reviews_from_json_gz(data_path)
    
    # Use larger sample for better performance
    if len(df) > sample_size:
        # Stratified sampling to preserve user distribution
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 5].index
        
        # Sample more from active users
        active_df = df[df['user_id'].isin(active_users)]
        other_df = df[~df['user_id'].isin(active_users)]
        
        n_active = min(len(active_df), int(sample_size * 0.7))
        n_other = min(len(other_df), sample_size - n_active)
        
        sampled_df = pd.concat([
            active_df.sample(n=n_active, random_state=DEFAULT_RANDOM_STATE),
            other_df.sample(n=n_other, random_state=DEFAULT_RANDOM_STATE)
        ])
        df = sampled_df
        
    print(f"Working with {len(df)} interactions from {df['user_id'].nunique()} users")
    
    # Enhanced preprocessing
    df = enhanced_preprocess_data(df, preserve_percentage=0.85)
    
    # Stratified train-test split
    train_df, test_df = EnhancedEvaluator.stratified_train_test_split(df, test_ratio=0.2)
    
    print(f"\nTrain: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    print(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")
    
    # Optimized configurations for 5-10% performance
    configs = [
        {
            "name": "Optimal-Enhanced-Cosine",
            "similarity_method": "enhanced_cosine",
            "k": 80,  # More neighbors for sparse data
            "min_similarity": 0.00001,  # Very low threshold
            "min_interactions": 1,
            "popularity_weight": 0.35,  # Higher popularity weight
            "time_decay_factor": 0.9,
            "use_idf_weighting": True
        },
        {
            "name": "High-Coverage-Tanimoto",
            "similarity_method": "tanimoto",
            "k": 60,
            "min_similarity": 0.00001,
            "min_interactions": 1,
            "popularity_weight": 0.4,
            "time_decay_factor": 0.85,
            "use_idf_weighting": False
        },
        {
            "name": "Balanced-Jaccard-Cooccur",
            "similarity_method": "jaccard",
            "k": 70,
            "min_similarity": 0.00001,
            "min_interactions": 1,
            "popularity_weight": 0.3,
            "time_decay_factor": 0.95,
            "use_idf_weighting": False
        }
    ]
    
    best_result = None
    best_config = None
    best_model = None
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*50}")
        
        try:
            model = EnhancedItemToItemRecommender(**{k: v for k, v in config.items() if k != 'name'})
            model.fit(train_df, timestamp_col='timestamp')
            
            # Comprehensive evaluation
            results = EnhancedEvaluator.comprehensive_evaluation(model, test_df, train_df, k_values=k_values)
            
            # Display results
            print("\nMetrics:")
            for k in k_values:
                print(f"\n@{k}:")
                for metric in ['Precision', 'Recall', 'F1', 'NDCG', 'HitRate']:
                    key = f"{metric}@{k}"
                    if key in results:
                        print(f"  {metric}: {results[key]:.4f}")
            
            # Track best model
            avg_f1 = np.mean([results.get(f"F1@{k}", 0) for k in k_values])
            if best_result is None or avg_f1 > np.mean([best_result.get(f"F1@{k}", 0) for k in k_values]):
                best_result = results
                best_config = config
                best_model = model
                
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if best_result:
        print(f"\nBest Configuration: {best_config['name']}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['Precision', 'Recall', 'F1', 'NDCG', 'HitRate']:
            values = [best_result.get(f"{metric}@{k}", 0) for k in k_values]
            avg_metrics[metric] = np.mean(values)
        
        print("\nAverage Performance Across K-values:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f} ({value*100:.1f}%)")
        
        # Check if target achieved
        if avg_metrics['F1'] >= 0.05:
            print("\n✅ SUCCESS: Achieved 5%+ F1 score!")
        else:
            improvement_needed = 0.05 / avg_metrics['F1'] if avg_metrics['F1'] > 0 else float('inf')
            print(f"\n📈 Progress: {avg_metrics['F1']*100:.1f}% (need {improvement_needed:.1f}x improvement)")
        
        # Save best model
        save_model(best_model, best_config, best_result)
        
        # Generate sample recommendations
        print_sample_recommendations(best_model, train_df, test_df)
    
    return best_model, best_result

def save_model(model, config, results):
    """Save the best model for later use"""
    try:
        with open('best_cf_model.pkl', 'wb') as f:
            pickle.dump({
                'model': model,
                'config': config,
                'results': results
            }, f)
        print("\n💾 Model saved to 'best_cf_model.pkl'")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")

def print_sample_recommendations(model, train_df, test_df):
    """Print sample recommendations for analysis"""
    print(f"\n{'='*60}")
    print("SAMPLE RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Get a few test users
    test_users = test_df['user_id'].unique()[:5]
    
    for user in test_users:
        user_history = train_df[train_df['user_id'] == user]['item_id'].tolist()
        if len(user_history) < 3:
            continue
            
        print(f"\nUser: {user}")
        print(f"History ({len(user_history)} items): {user_history[:5]}...")
        
        recs = model.recommend_items(user_history, top_n=5)
        print("Recommendations:")
        for i, (item, score) in enumerate(recs, 1):
            print(f"  {i}. {item} (score: {score:.4f})")

def run_ensemble_evaluation(data_path=None, sample_size=DEFAULT_SAMPLE_SIZE):
    """Run ensemble of multiple CF approaches"""
    if data_path is None:
        data_path = "/home/zalert_rig305/Desktop/EE/Programs/Clothing_Shoes_and_Jewelry.json.gz"
    
    print("="*60)
    print("ENSEMBLE CF EVALUATION")
    print("="*60)
    
    # Load and preprocess data
    df = load_reviews_from_json_gz(data_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    df = enhanced_preprocess_data(df)
    train_df, test_df = EnhancedEvaluator.stratified_train_test_split(df)
    
    # Train multiple models with different approaches
    models = []
    
    # Model 1: Enhanced Cosine with high k
    model1 = EnhancedItemToItemRecommender(
        k=100, similarity_method='enhanced_cosine', 
        popularity_weight=0.3, use_idf_weighting=True
    )
    model1.fit(train_df, timestamp_col='timestamp')
    models.append(('Enhanced-Cosine', model1, 0.4))
    
    # Model 2: Jaccard with co-occurrence
    model2 = EnhancedItemToItemRecommender(
        k=80, similarity_method='jaccard',
        popularity_weight=0.35, use_idf_weighting=False
    )
    model2.fit(train_df, timestamp_col='timestamp')
    models.append(('Jaccard-Cooccur', model2, 0.3))
    
    # Model 3: Tanimoto with temporal
    model3 = EnhancedItemToItemRecommender(
        k=60, similarity_method='tanimoto',
        popularity_weight=0.4, time_decay_factor=0.85
    )
    model3.fit(train_df, timestamp_col='timestamp')
    models.append(('Tanimoto-Temporal', model3, 0.3))
    
    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    ensemble_results = evaluate_ensemble(models, test_df, train_df)
    
    print("\nEnsemble Results:")
    for metric, value in ensemble_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
    
    return ensemble_results

def evaluate_ensemble(models, test_df, train_df, k=10):
    """Evaluate ensemble of models"""
    precisions = []
    recalls = []
    f1_scores = []
    hits = 0
    total_users = 0
    
    test_grouped = test_df.groupby('user_id')
    
    for user, test_items_df in test_grouped:
        test_items = set(test_items_df['item_id'].tolist())
        user_history = train_df[train_df['user_id'] == user]['item_id'].tolist()
        
        if len(user_history) < 1:
            continue
        
        # Get recommendations from each model
        ensemble_scores = defaultdict(float)
        
        for name, model, weight in models:
            try:
                recs = model.recommend_items(user_history, top_n=k*2)  # Get more candidates
                for item, score in recs:
                    ensemble_scores[item] += score * weight
            except:
                continue
        
        if not ensemble_scores:
            continue
        
        # Get top-k from ensemble
        ensemble_recs = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rec_items = [item for item, _ in ensemble_recs]
        
        # Calculate metrics
        relevant_retrieved = len(set(rec_items).intersection(test_items))
        
        if relevant_retrieved > 0:
            hits += 1
            precision = relevant_retrieved / len(rec_items)
            recall = relevant_retrieved / len(test_items)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        total_users += 1
    
    return {
        f"Ensemble_Precision@{k}": np.mean(precisions) if precisions else 0,
        f"Ensemble_Recall@{k}": np.mean(recalls) if recalls else 0,
        f"Ensemble_F1@{k}": np.mean(f1_scores) if f1_scores else 0,
        f"Ensemble_HitRate@{k}": hits / total_users if total_users > 0 else 0,
        "Ensemble_Users_Evaluated": total_users
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced CF Evaluation")
    parser.add_argument('--mode', choices=['single', 'ensemble'], default='single')
    parser.add_argument('--sample_size', type=int, default=200000)
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 20])
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_optimized_evaluation(sample_size=args.sample_size, k_values=args.top_k)
    else:
        run_ensemble_evaluation(sample_size=args.sample_size)