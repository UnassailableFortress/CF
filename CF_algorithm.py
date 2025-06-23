import pandas as pd
import gzip
import json
from tqdm import tqdm

from CF_Module import ItemToItemRecommender  # Your refactored class
from Dataloader import create_efficient_dataloader

def load_reviews_from_json_gz(path):
    """
    Loads Amazon review data from a JSON.gz file into a pandas DataFrame.
    """
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {path}"):
            data = json.loads(line)
            rows.append({
                'user_id': data['reviewerID'],
                'item_id': data['asin'],
                'rating': data.get('overall', 1.0)
            })
    return pd.DataFrame(rows)

def load_and_concat(file_map, base_path):
    """
    Loads multiple category files and concatenates them.
    """
    dfs = []
    for category, filename in file_map.items():
        path = f"{base_path}/Movies_and_TV.json.gz"
        df = load_reviews_from_json_gz(path)
        df['category'] = category
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    base_path = "/home/zalert_rig305/Desktop/EE/Programs"
    file_map = {
        "Electronics": "Electronics.json.gz",
        "Books": "Books.json.gz",
        "Movies_and_TV": "Movies_and_TV.json.gz",
        "Clothing_Shoes_and_Jewelry": "Clothing_Shoes_and_Jewelry.json.gz"
    }

    all_reviews_df = load_and_concat(file_map, base_path)
    print("Combined review dataset shape:", all_reviews_df.shape)

    # OPTIONAL: Limit data for initial testing
    # all_reviews_df = all_reviews_df.sample(n=100_000, random_state=42)

    model = ItemToItemRecommender(k=10)
    model.fit(all_reviews_df, user_col='user_id', item_col='item_id', rating_col='rating')

    # Example usage
    example_user = 'A2BFF1TLPUX9DJ'
    user_history = all_reviews_df[all_reviews_df['user_id'] == example_user]['item_id'].unique().tolist()
    recs = model.recommend_items(user_history, top_n=10)
    print("Recommendations for user:", recs)
