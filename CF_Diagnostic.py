"""
CF_Diagnostics.py - Comprehensive Diagnostic Tool for CF System Issues
Identifies root causes of poor performance metrics and provides actionable solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
import gzip
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder

class CFDiagnosticTool:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.issues_found = []
        self.recommendations = []
        self.stats = {}

    def run_full_diagnosis(self, data_path, sample_size=100000):
        """Run complete diagnostic analysis"""
        print("="*70)
        print("COLLABORATIVE FILTERING SYSTEM DIAGNOSTIC TOOL")
        print("="*70)
        
        # Load and sample data
        self.data = self._load_data(data_path, sample_size)
        if self.data is None or len(self.data) == 0:
            print("ERROR: No data loaded!")
            return
        
        print(f"Analyzing {len(self.data)} interactions...")
        
        # Run all diagnostic checks
        self._check_data_quality()
        self._check_sparsity_issues()
        self._check_distribution_problems()
        self._analyze_user_item_overlap()
        self._check_train_test_split_issues()
        self._analyze_similarity_potential()
        self._check_cold_start_problems()
        
        # Generate actionable recommendations
        self._generate_recommendations()
        
        # Print comprehensive report
        self._print_diagnostic_report()
        
        return self.issues_found, self.recommendations

    def _load_data(self, path, sample_size=None):
        """Load and optionally sample data"""
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return None
        
        print(f"Loading data from {path}...")
        rows = []
        
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    try:
                        data = json.loads(line)
                        rows.append({
                            'user_id': data.get('reviewerID', 'unknown'),
                            'item_id': data.get('asin', 'unknown'),
                            'rating': float(data.get('overall', 1.0)),
                            'timestamp': data.get('unixReviewTime', 0)
                        })
                    except:
                        continue
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return None
        
        df = pd.DataFrame(rows)
        # Basic cleaning
        df = df[(df['user_id'] != 'unknown') & (df['item_id'] != 'unknown')]
        
        print(f"Loaded {len(df)} valid interactions")
        return df

    def _check_data_quality(self):
        """Check basic data quality issues"""
        print("\n" + "="*50)
        print("1. DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Basic statistics
        n_users = self.data['user_id'].nunique()
        n_items = self.data['item_id'].nunique()
        n_interactions = len(self.data)
        
        self.stats['n_users'] = n_users
        self.stats['n_items'] = n_items
        self.stats['n_interactions'] = n_interactions
        
        print(f"Users: {n_users:,}")
        print(f"Items: {n_items:,}")
        print(f"Interactions: {n_interactions:,}")
        
        # Data quality checks
        duplicate_interactions = self.data.duplicated(['user_id', 'item_id']).sum()
        if duplicate_interactions > 0:
            self.issues_found.append(f"Found {duplicate_interactions} duplicate user-item interactions")
            print(f"⚠️  ISSUE: {duplicate_interactions} duplicate interactions found")
        
        # Rating distribution
        rating_dist = self.data['rating'].value_counts().sort_index()
        print(f"\nRating distribution:")
        for rating, count in rating_dist.items():
            print(f"  {rating}: {count:,} ({count/len(self.data)*100:.1f}%)")
        
        # Check for rating bias
        if rating_dist.get(5.0, 0) / len(self.data) > 0.6:
            self.issues_found.append("High rating bias detected (>60% ratings are 5.0)")
            print("⚠️  ISSUE: Severe positive rating bias detected")

    def _check_sparsity_issues(self):
        """Analyze sparsity problems"""
        print("\n" + "="*50)
        print("2. SPARSITY ANALYSIS")
        print("="*50)
        
        n_users = self.stats['n_users']
        n_items = self.stats['n_items']
        n_interactions = self.stats['n_interactions']
        
        # Calculate sparsity
        total_possible = n_users * n_items
        sparsity = 1 - (n_interactions / total_possible)
        density = n_interactions / total_possible
        
        self.stats['sparsity'] = sparsity
        self.stats['density'] = density
        
        print(f"Matrix size: {n_users:,} × {n_items:,} = {total_possible:,} possible interactions")
        print(f"Actual interactions: {n_interactions:,}")
        print(f"Density: {density:.6f} ({density*100:.4f}%)")
        print(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
        
        # Sparsity severity assessment
        if sparsity > 0.999:
            self.issues_found.append(f"CRITICAL: Extreme sparsity ({sparsity:.6f})")
            print("🚨 CRITICAL: Extreme sparsity detected!")
        elif sparsity > 0.99:
            self.issues_found.append(f"HIGH: Very high sparsity ({sparsity:.6f})")
            print("⚠️  HIGH: Very high sparsity")
        
        # Analyze user interaction patterns
        user_interactions = self.data['user_id'].value_counts()
        item_interactions = self.data['item_id'].value_counts()
        
        print(f"\nUser interaction statistics:")
        print(f"  Mean: {user_interactions.mean():.2f}")
        print(f"  Median: {user_interactions.median():.2f}")
        print(f"  Min: {user_interactions.min()}")
        print(f"  Max: {user_interactions.max()}")
        
        print(f"\nItem interaction statistics:")
        print(f"  Mean: {item_interactions.mean():.2f}")
        print(f"  Median: {item_interactions.median():.2f}")
        print(f"  Min: {item_interactions.min()}")
        print(f"  Max: {item_interactions.max()}")
        
        # Check for power-law distribution issues
        users_with_few_interactions = (user_interactions <= 2).sum()
        items_with_few_interactions = (item_interactions <= 2).sum()
        
        if users_with_few_interactions / n_users > 0.8:
            self.issues_found.append(f"Too many users with ≤2 interactions ({users_with_few_interactions/n_users:.1%})")
            print(f"⚠️  ISSUE: {users_with_few_interactions/n_users:.1%} users have ≤2 interactions")
        
        if items_with_few_interactions / n_items > 0.8:
            self.issues_found.append(f"Too many items with ≤2 interactions ({items_with_few_interactions/n_items:.1%})")
            print(f"⚠️  ISSUE: {items_with_few_interactions/n_items:.1%} items have ≤2 interactions")

    def _check_distribution_problems(self):
        """Check for distribution issues that affect CF performance"""
        print("\n" + "="*50)
        print("3. DISTRIBUTION ANALYSIS")
        print("="*50)
        
        user_interactions = self.data['user_id'].value_counts()
        item_interactions = self.data['item_id'].value_counts()
        
        # Check for extreme popularity bias
        top_1_percent_items = int(len(item_interactions) * 0.01)
        top_items_interactions = item_interactions.head(top_1_percent_items).sum()
        popularity_concentration = top_items_interactions / self.stats['n_interactions']
        
        print(f"Top 1% items account for {popularity_concentration:.1%} of all interactions")
        
        if popularity_concentration > 0.5:
            self.issues_found.append(f"Severe popularity bias: top 1% items = {popularity_concentration:.1%} interactions")
            print("🚨 CRITICAL: Severe popularity bias detected!")
        
        # Check user activity distribution
        top_1_percent_users = int(len(user_interactions) * 0.01)
        top_users_interactions = user_interactions.head(top_1_percent_users).sum()
        user_concentration = top_users_interactions / self.stats['n_interactions']
        
        print(f"Top 1% users account for {user_concentration:.1%} of all interactions")
        
        if user_concentration > 0.3:
            self.issues_found.append(f"High user activity concentration: top 1% users = {user_concentration:.1%} interactions")
            print("⚠️  ISSUE: High user activity concentration")
        
        # Gini coefficient for inequality measurement
        user_gini = self._calculate_gini(user_interactions.values)
        item_gini = self._calculate_gini(item_interactions.values)
        
        print(f"\nGini coefficients (0=equal, 1=completely unequal):")
        print(f"  User interactions: {user_gini:.3f}")
        print(f"  Item interactions: {item_gini:.3f}")
        
        if user_gini > 0.8:
            self.issues_found.append(f"Very unequal user distribution (Gini: {user_gini:.3f})")
        if item_gini > 0.8:
            self.issues_found.append(f"Very unequal item distribution (Gini: {item_gini:.3f})")

    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _analyze_user_item_overlap(self):
        """Analyze overlap patterns that affect similarity computation"""
        print("\n" + "="*50)
        print("4. USER-ITEM OVERLAP ANALYSIS")
        print("="*50)
        
        # Sample analysis for performance
        sample_users = self.data['user_id'].unique()[:1000]  # Sample first 1000 users
        
        overlap_counts = []
        similarity_potential = []
        
        print("Analyzing user-item overlaps (sample of 1000 users)...")
        
        for i, user1 in enumerate(sample_users[:100]):  # Check first 100 users
            user1_items = set(self.data[self.data['user_id'] == user1]['item_id'])
            
            for user2 in sample_users[i+1:i+51]:  # Compare with next 50 users
                user2_items = set(self.data[self.data['user_id'] == user2]['item_id'])
                
                overlap = len(user1_items.intersection(user2_items))
                union = len(user1_items.union(user2_items))
                
                if union > 0:
                    jaccard_sim = overlap / union
                    overlap_counts.append(overlap)
                    similarity_potential.append(jaccard_sim)
        
        if overlap_counts:
            avg_overlap = np.mean(overlap_counts)
            avg_similarity = np.mean(similarity_potential)
            
            print(f"Average item overlap between users: {avg_overlap:.2f}")
            print(f"Average Jaccard similarity: {avg_similarity:.4f}")
            
            if avg_overlap < 1.0:
                self.issues_found.append(f"Very low user-item overlap (avg: {avg_overlap:.2f})")
                print("🚨 CRITICAL: Very low user-item overlap!")
            
            if avg_similarity < 0.01:
                self.issues_found.append(f"Extremely low user similarity potential ({avg_similarity:.4f})")
                print("🚨 CRITICAL: Extremely low similarity potential!")

    def _check_train_test_split_issues(self):
        """Analyze train/test split problems"""
        print("\n" + "="*50)
        print("5. TRAIN/TEST SPLIT ANALYSIS")
        print("="*50)
        
        # Simulate leave-one-out split
        users_with_multiple_interactions = self.data['user_id'].value_counts()
        users_for_eval = users_with_multiple_interactions[users_with_multiple_interactions >= 2]
        
        print(f"Users with ≥2 interactions (evaluatable): {len(users_for_eval)}")
        print(f"Total users: {self.stats['n_users']}")
        print(f"Evaluation coverage: {len(users_for_eval)/self.stats['n_users']:.1%}")
        
        if len(users_for_eval) / self.stats['n_users'] < 0.3:
            self.issues_found.append(f"Low evaluation coverage: only {len(users_for_eval)/self.stats['n_users']:.1%} users evaluatable")
            print("⚠️  ISSUE: Low evaluation coverage")
        
        # Check for temporal bias if timestamps available
        if 'timestamp' in self.data.columns:
            timestamps = pd.to_datetime(self.data['timestamp'], unit='s')
            time_span = (timestamps.max() - timestamps.min()).days
            print(f"Dataset time span: {time_span} days")
            
            if time_span < 30:
                self.issues_found.append(f"Very short time span ({time_span} days) - may lack temporal diversity")
                print("⚠️  ISSUE: Very short time span")

    def _analyze_similarity_potential(self):
        """Analyze potential for similarity computation"""
        print("\n" + "="*50)
        print("6. SIMILARITY COMPUTATION POTENTIAL")
        print("="*50)
        
        # Sample items for analysis
        sample_items = self.data['item_id'].unique()[:500]  # Sample 500 items
        
        item_user_counts = {}
        for item in sample_items:
            users = set(self.data[self.data['item_id'] == item]['user_id'])
            item_user_counts[item] = len(users)
        
        # Analyze potential for similarity computation
        items_with_multiple_users = sum(1 for count in item_user_counts.values() if count >= 2)
        
        print(f"Items with ≥2 users (similarity computable): {items_with_multiple_users}/{len(sample_items)}")
        print(f"Similarity computation potential: {items_with_multiple_users/len(sample_items):.1%}")
        
        if items_with_multiple_users / len(sample_items) < 0.5:
            self.issues_found.append(f"Low similarity potential: only {items_with_multiple_users/len(sample_items):.1%} items have ≥2 users")
            print("🚨 CRITICAL: Low similarity computation potential!")
        
        # Check for co-occurrence patterns
        sample_pairs = 0
        co_occurrence_count = 0
        
        for i, item1 in enumerate(sample_items[:50]):
            for item2 in sample_items[i+1:i+21]:  # Check 20 pairs per item
                users1 = set(self.data[self.data['item_id'] == item1]['user_id'])
                users2 = set(self.data[self.data['item_id'] == item2]['user_id'])
                
                if len(users1.intersection(users2)) > 0:
                    co_occurrence_count += 1
                sample_pairs += 1
        
        if sample_pairs > 0:
            co_occurrence_rate = co_occurrence_count / sample_pairs
            print(f"Item co-occurrence rate: {co_occurrence_rate:.3f}")
            
            if co_occurrence_rate < 0.1:
                self.issues_found.append(f"Very low item co-occurrence rate ({co_occurrence_rate:.3f})")
                print("⚠️  ISSUE: Very low item co-occurrence")

    def _check_cold_start_problems(self):
        """Analyze cold start issues"""
        print("\n" + "="*50)
        print("7. COLD START ANALYSIS")
        print("="*50)
        
        user_interactions = self.data['user_id'].value_counts()
        item_interactions = self.data['item_id'].value_counts()
        
        # Cold start users (≤1 interaction)
        cold_start_users = (user_interactions <= 1).sum()
        cold_start_user_pct = cold_start_users / len(user_interactions)
        
        # Cold start items (≤1 interaction)  
        cold_start_items = (item_interactions <= 1).sum()
        cold_start_item_pct = cold_start_items / len(item_interactions)
        
        print(f"Cold start users (≤1 interaction): {cold_start_users} ({cold_start_user_pct:.1%})")
        print(f"Cold start items (≤1 interaction): {cold_start_items} ({cold_start_item_pct:.1%})")
        
        if cold_start_user_pct > 0.5:
            self.issues_found.append(f"High cold-start user ratio: {cold_start_user_pct:.1%}")
            print("⚠️  ISSUE: High cold-start user ratio")
        
        if cold_start_item_pct > 0.5:
            self.issues_found.append(f"High cold-start item ratio: {cold_start_item_pct:.1%}")
            print("⚠️  ISSUE: High cold-start item ratio")

    def _generate_recommendations(self):
        """Generate actionable recommendations based on found issues"""
        print("\n" + "="*50)
        print("8. GENERATING RECOMMENDATIONS")
        print("="*50)
        
        # Data filtering recommendations
        if any("sparsity" in issue.lower() for issue in self.issues_found):
            self.recommendations.append({
                "category": "Data Filtering",
                "action": "Increase minimum interaction thresholds",
                "details": "Filter users with <5 interactions and items with <5 interactions",
                "code": "df = df[df.groupby('user_id')['user_id'].transform('count') >= 5]"
            })
        
        # Similarity threshold recommendations
        if any("overlap" in issue.lower() or "similarity" in issue.lower() for issue in self.issues_found):
            self.recommendations.append({
                "category": "Similarity Computation", 
                "action": "Lower similarity thresholds and increase k",
                "details": "Use min_similarity=0.001, k=50-100, try Jaccard for binary data",
                "code": "model = EnhancedItemToItemRecommender(k=50, min_similarity=0.001, similarity_method='jaccard')"
            })
        
        # Evaluation recommendations
        if any("coverage" in issue.lower() or "evaluation" in issue.lower() for issue in self.issues_found):
            self.recommendations.append({
                "category": "Evaluation Strategy",
                "action": "Improve train/test split strategy",
                "details": "Use temporal split or stratified sampling, require min 3 interactions for evaluation",
                "code": "train_df, test_df = temporal_split(df, test_ratio=0.2, min_interactions=3)"
            })
        
        # Popularity bias recommendations
        if any("popularity" in issue.lower() or "bias" in issue.lower() for issue in self.issues_found):
            self.recommendations.append({
                "category": "Popularity Bias",
                "action": "Add popularity-based regularization",
                "details": "Penalize popular items in recommendations, boost long-tail items",
                "code": "popularity_penalty = np.log(1 + item_popularity[item]) * 0.1"
            })
        
        # Cold start recommendations
        if any("cold" in issue.lower() for issue in self.issues_found):
            self.recommendations.append({
                "category": "Cold Start Handling",
                "action": "Implement hybrid approach",
                "details": "Combine collaborative filtering with content-based and popularity-based methods",
                "code": "final_score = 0.7 * cf_score + 0.2 * content_score + 0.1 * popularity_score"
            })

    def _print_diagnostic_report(self):
        """Print comprehensive diagnostic report"""
        print("\n" + "="*70)
        print("DIAGNOSTIC REPORT SUMMARY")
        print("="*70)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Users: {self.stats.get('n_users', 0):,}")
        print(f"   Items: {self.stats.get('n_items', 0):,}")
        print(f"   Interactions: {self.stats.get('n_interactions', 0):,}")
        print(f"   Sparsity: {self.stats.get('sparsity', 0):.6f}")
        
        print(f"\n🚨 ISSUES FOUND ({len(self.issues_found)}):")
        if self.issues_found:
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("   No critical issues detected!")
        
        print(f"\n💡 RECOMMENDATIONS ({len(self.recommendations)}):")
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"\n   {i}. {rec['category']}: {rec['action']}")
                print(f"      {rec['details']}")
                print(f"      Code: {rec['code']}")
        else:
            print("   No specific recommendations generated.")
        
        print(f"\n🎯 NEXT STEPS:")
        print("   1. Implement data filtering recommendations")
        print("   2. Adjust algorithm parameters based on diagnostics")
        print("   3. Consider hybrid approaches for cold-start problems")
        print("   4. Re-run evaluation with improved configuration")
        
        print("\n" + "="*70)

    def save_report(self, filename="cf_diagnostic_report.txt"):
        """Save diagnostic report to file"""
        with open(filename, 'w') as f:
            f.write("COLLABORATIVE FILTERING DIAGNOSTIC REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("DATASET STATISTICS:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nISSUES FOUND ({len(self.issues_found)}):\n")
            for i, issue in enumerate(self.issues_found, 1):
                f.write(f"  {i}. {issue}\n")
            
            f.write(f"\nRECOMMENDATIONS ({len(self.recommendations)}):\n")
            for i, rec in enumerate(self.recommendations, 1):
                f.write(f"\n  {i}. {rec['category']}: {rec['action']}\n")
                f.write(f"     {rec['details']}\n")
                f.write(f"     Code: {rec['code']}\n")
        
        print(f"Report saved to {filename}")


def run_diagnostics(data_path=None, sample_size=100000):
    """Main function to run CF diagnostics"""
    if data_path is None:
        data_path = "/home/zalert_rig305/Desktop/EE/Programs/Movies_and_TV.json.gz"
    
    diagnostic_tool = CFDiagnosticTool()
    issues, recommendations = diagnostic_tool.run_full_diagnosis(data_path, sample_size)
    
    # Save report
    diagnostic_tool.save_report("cf_diagnostic_report.txt")
    
    return diagnostic_tool


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CF System Diagnostics")
    parser.add_argument('--data_path', type=str, default=None, help="Path to data file")
    parser.add_argument('--sample_size', type=int, default=100000, help="Sample size for analysis")
    
    args = parser.parse_args()
    
    diagnostic_tool = run_diagnostics(args.data_path, args.sample_size)