import pandas as pd
import numpy as np
from CF_Module import FixedItemToItemRecommender, FixedEvaluator
from Dataloader import create_efficient_dataloader
import gzip
import json
from tqdm import tqdm
import glob
import os

# Configuration constants
DEFAULT_SAMPLE_SIZE = 100000
DEFAULT_TOP_K = 10
DEFAULT_RANDOM_STATE = 42
MIN_ROWS_THRESHOLD = 5000
MAX_PREPROCESSING_ITERATIONS = 5

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
                    print(f"Warning: Skipping malformed JSON line: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise

    return pd.DataFrame(rows)

def preprocess_data(df, min_user_interactions=1, min_item_interactions=1,
                   max_iterations=MAX_PREPROCESSING_ITERATIONS,
                   min_rows_threshold=MIN_ROWS_THRESHOLD):
    if df.empty:
        raise ValueError("Input dataframe is empty")

    print(f"Initial shape: {df.shape}")
    df = df[(df['user_id'] != 'unknown') & (df['item_id'] != 'unknown')]
    print(f"After removing unknowns: {df.shape}")

    if df.empty:
        raise ValueError("No valid data after removing unknown users/items")

    for iteration in range(max_iterations):
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        prev_shape = df.shape[0]

        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        df = df[df['user_id'].isin(valid_users)]
        df = df[df['item_id'].isin(valid_items)]

        print(f"Iteration {iteration+1}: {df.shape}")

        if df.shape[0] < min_rows_threshold:
            print(f"Warning: Too few rows after filtering ({df.shape[0]}). Stopping early.")
            break

        if df.shape[0] == prev_shape:
            print("Preprocessing converged.")
            break

    return df

def run_enhanced_evaluation(data_path=None, sample_size=DEFAULT_SAMPLE_SIZE, top_k=DEFAULT_TOP_K):
    if data_path is None:
        data_path = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz"

    print("="*60)
    print("ENHANCED COLLABORATIVE FILTERING EVALUATION - TUNING MODE")
    print("="*60)
    print("\n1. LOADING AND PREPROCESSING DATA")

    try:
        df = load_reviews_from_json_gz(data_path)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=DEFAULT_RANDOM_STATE)
            print(f"Sampled {sample_size} records from {len(df)} total")
        
        df = preprocess_data(df)
        if df.empty or len(df) < 100:
            raise ValueError("Dataset too small after preprocessing.")
        
        # Use the correct method name from FixedEvaluator
        train_df, test_df = FixedEvaluator.gentle_train_test_split(df, min_train_interactions=2)
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return

    param_grid = [
        {"similarity_method": "cosine", "k": 20, "min_similarity": 0.001, "min_interactions": 1},
        {"similarity_method": "cosine", "k": 30, "min_similarity": 0.001, "min_interactions": 2},
        {"similarity_method": "jaccard", "k": 15, "min_similarity": 0.001, "min_interactions": 1},
    ]

    best_result = None
    best_config = None

    print(f"\n2. RUNNING PARAMETER TUNING ({len(param_grid)} configurations)")

    for i, params in enumerate(param_grid):
        print(f"\nRunning config {i+1}/{len(param_grid)}: {params}")

        try:
            # Use FixedItemToItemRecommender instead of EnhancedItemToItemRecommender
            model = FixedItemToItemRecommender(
                k=params["k"],
                similarity_method=params["similarity_method"],
                min_similarity=params["min_similarity"],
                min_interactions=params["min_interactions"],
                implicit_feedback=True
            )

            model.fit(train_df, timestamp_col='timestamp', category_col='category')
            
            # Use robust_evaluation from FixedEvaluator
            results = FixedEvaluator.robust_evaluation(model, test_df, train_df, k=top_k)

            print("\nResults:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

            # Compare based on F1 score (which is available in the results)
            if best_result is None or results[f"F1@{top_k}"] > best_result[f"F1@{top_k}"]:
                best_result = results
                best_config = params

        except Exception as e:
            print(f"Error in configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    if best_result and best_config:
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")

        print("\nBest Results:")
        for metric, value in best_result.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        print("No successful configurations found.")

def run_chunked_evaluation(chunk_dir=None, top_k=DEFAULT_TOP_K):
    """Run evaluation on individual chunks separately"""
    if chunk_dir is None:
        chunk_dir = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV_Split/"

    if not os.path.exists(chunk_dir):
        print(f"Error: Directory {chunk_dir} does not exist")
        return

    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "part*.json")))
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files")
    
    all_results = []

    for i, file_path in enumerate(chunk_files):
        print(f"\n==== Processing Chunk {i+1}/{len(chunk_files)}: {os.path.basename(file_path)} ====")

        try:
            # Load chunk
            df = pd.read_json(file_path, lines=True)
            df = df.rename(columns={
                'reviewerID': 'user_id',
                'asin': 'item_id',
                'overall': 'rating',
                'unixReviewTime': 'timestamp'
            })
            df['category'] = 'Movies_and_TV'
            
            # Preprocess
            df = preprocess_data(df)
            if df.empty or len(df) < 100:
                print("Chunk too small after preprocessing, skipping.")
                continue
            
            # Split data
            train_df, test_df = FixedEvaluator.gentle_train_test_split(df)
            
            if len(test_df) == 0:
                print("No test data available for this chunk, skipping.")
                continue
                
            print(f"Chunk {i+1} - Train: {len(train_df)}, Test: {len(test_df)}")
            
            # Train model
            model = FixedItemToItemRecommender(
                k=20,
                similarity_method='cosine',
                min_similarity=0.001,
                min_interactions=1,
                implicit_feedback=True
            )
            
            model.fit(train_df, timestamp_col='timestamp', category_col='category')
            results = FixedEvaluator.robust_evaluation(model, test_df, train_df, k=top_k)
            
            print(f"\nChunk {i+1} Results:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
                    
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing chunk {file_path}: {e}")
            continue

    # Summarize results across all chunks
    if all_results:
        print("\n==== SUMMARY ACROSS ALL CHUNKS ====")
        metric_names = list(all_results[0].keys())
        
        for metric in metric_names:
            if isinstance(all_results[0][metric], (int, float)):
                values = [r[metric] for r in all_results if metric in r]
                if values:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    print(f"{metric}: {avg_value:.4f} (±{std_value:.4f})")

def run_continuous_training_across_chunks(chunk_dir=None, top_k=DEFAULT_TOP_K):
    """Accumulate data from chunks and train on combined dataset"""
    if chunk_dir is None:
        chunk_dir = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV_Split/"

    if not os.path.exists(chunk_dir):
        print(f"Error: Directory {chunk_dir} does not exist")
        return

    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "part*.json")))
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files")
    combined_df = []

    for i, file_path in enumerate(chunk_files):
        print(f"\n==== Loading Chunk {i+1}/{len(chunk_files)}: {os.path.basename(file_path)} ====")

        try:
            df = pd.read_json(file_path, lines=True)
            df = df.rename(columns={
                'reviewerID': 'user_id',
                'asin': 'item_id',
                'overall': 'rating',
                'unixReviewTime': 'timestamp'
            })
            df['category'] = 'Movies_and_TV'
            df = preprocess_data(df)
            if df.empty or len(df) < 100:
                print("Chunk too small after preprocessing, skipping.")
                continue
            combined_df.append(df)
        except Exception as e:
            print(f"Error loading chunk {file_path}: {e}")
            continue

    if not combined_df:
        print("No valid data loaded from chunks.")
        return

    full_df = pd.concat(combined_df, ignore_index=True)
    print(f"\nCombined dataset shape: {full_df.shape}")

    train_df, test_df = FixedEvaluator.gentle_train_test_split(full_df)
    print(f"Train interactions: {len(train_df)} | Test interactions: {len(test_df)}")

    model = FixedItemToItemRecommender(
        k=20,
        similarity_method='cosine',
        min_similarity=0.001,
        min_interactions=1,
        implicit_feedback=True
    )

    model.fit(train_df, timestamp_col='timestamp', category_col='category')
    results = FixedEvaluator.robust_evaluation(model, test_df, train_df, k=top_k)

    print("\n==== FINAL EVALUATION ON COMBINED DATA ====")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Collaborative Filtering Evaluation")
    parser.add_argument('--mode', type=str, choices=['single', 'chunked', 'continuous'], default='single',
                        help="Evaluation mode: 'single' for one dataset, 'chunked' for separate models, 'continuous' for accumulating training")
    parser.add_argument('--top_k', type=int, default=10, help="Top-K for evaluation")
    parser.add_argument('--chunk_dir', type=str, default=None, help="Directory of chunk files")

    args = parser.parse_args()

    if args.mode == 'single':
        run_enhanced_evaluation()
    elif args.mode == 'chunked':
        run_chunked_evaluation(chunk_dir=args.chunk_dir, top_k=args.top_k)
    elif args.mode == 'continuous':
        run_continuous_training_across_chunks(chunk_dir=args.chunk_dir, top_k=args.top_k)
    else:
        print(f"Unsupported mode: {args.mode}")