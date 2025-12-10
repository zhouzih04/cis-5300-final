"""
Data Preparation for News Timeline Construction Pipeline

This script converts labeled timeline data (ETimeline format) into training datasets
for three stages:
    Stage 1: Pairwise relatedness classification (for XGBoost)
    Stage 2: Clustering constraints (for Leiden algorithm)
    Stage 3: Event threading/ordering classification

Author: Generated for ETimeline project
"""

import json
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path


# =============================================================================
# DATA LOADING
# =============================================================================

def load_labeled_data(filepath: str) -> List[Dict]:
    """
    Load labeled timeline data in ETimeline format.
    
    Expected format:
    [
        {
            "topic_index": "1",
            "topic": "North Korea launches...",
            "domain": "Military",
            "node_list": [
                {"node_index": "1-1", "title": "...", "date": "2023-05-31", "url": "..."},
                ...
            ]
        },
        ...
    ]
    """
    print(f"Loading labeled data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_articles = sum(len(topic['node_list']) for topic in data)
    print(f"Loaded {len(data)} topics with {total_articles} total articles")
    
    return data


def load_unlabeled_data(filepath: str) -> List[Dict]:
    """
    Load unlabeled article data.
    
    Expected format:
    [
        {"title": "...", "date": "2022-02-15", "url": "...", "language": "en"},
        ...
    ]
    """
    print(f"Loading unlabeled data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} unlabeled articles")
    return data


# =============================================================================
# STAGE 1: PAIRWISE CLASSIFICATION DATA
# =============================================================================

def generate_pairwise_data(
    topics: List[Dict],
    negative_ratio: float = 1.0,
    max_negatives_per_topic_pair: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate positive and negative pairs for pairwise classification.
    
    Positive pairs: Articles within the same topic
    Negative pairs: Articles from different topics (sampled)
    
    Args:
        topics: List of topic dictionaries with node_list
        negative_ratio: Ratio of negative to positive pairs (1.0 = balanced)
        max_negatives_per_topic_pair: Max negatives sampled per topic combination
        random_seed: For reproducibility
        
    Returns:
        DataFrame with columns: text_a, text_b, label, topic_a, topic_b, date_a, date_b
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print("Generating pairwise training data...")
    
    positive_pairs = []
    negative_pairs = []
    
    # Generate POSITIVE pairs (within same topic)
    for topic in topics:
        topic_idx = topic['topic_index']
        articles = topic['node_list']
        
        for art_a, art_b in combinations(articles, 2):
            positive_pairs.append({
                'text_a': art_a['title'],
                'text_b': art_b['title'],
                'label': 1,
                'topic_a': topic_idx,
                'topic_b': topic_idx,
                'date_a': art_a['date'],
                'date_b': art_b['date'],
                'node_a': art_a['node_index'],
                'node_b': art_b['node_index']
            })
    
    print(f"Generated {len(positive_pairs)} positive pairs")
    
    # Generate NEGATIVE pairs (across different topics)
    target_negatives = int(len(positive_pairs) * negative_ratio)
    topic_pairs = list(combinations(topics, 2))
    
    # Calculate how many negatives per topic pair
    negatives_per_pair = max(1, target_negatives // len(topic_pairs))
    negatives_per_pair = min(negatives_per_pair, max_negatives_per_topic_pair)
    
    for topic_a, topic_b in topic_pairs:
        articles_a = topic_a['node_list']
        articles_b = topic_b['node_list']
        
        # Sample articles from each topic
        sample_a = random.sample(articles_a, min(len(articles_a), negatives_per_pair))
        sample_b = random.sample(articles_b, min(len(articles_b), negatives_per_pair))
        
        for art_a in sample_a:
            for art_b in sample_b:
                negative_pairs.append({
                    'text_a': art_a['title'],
                    'text_b': art_b['title'],
                    'label': 0,
                    'topic_a': topic_a['topic_index'],
                    'topic_b': topic_b['topic_index'],
                    'date_a': art_a['date'],
                    'date_b': art_b['date'],
                    'node_a': art_a['node_index'],
                    'node_b': art_b['node_index']
                })
    
    # Balance if we have too many negatives
    if len(negative_pairs) > target_negatives:
        negative_pairs = random.sample(negative_pairs, target_negatives)
    
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    df = pd.DataFrame(all_pairs)
    print(f"Total pairwise dataset: {len(df)} pairs")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


# =============================================================================
# STAGE 2: CLUSTERING CONSTRAINTS
# =============================================================================

def generate_clustering_constraints(
    topics: List[Dict],
    sample_must_link: int = 1000,
    sample_cannot_link: int = 1000,
    random_seed: int = 42
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Generate must-link and cannot-link constraints for constrained clustering.
    
    Must-link: Pairs of articles that MUST be in the same cluster (same topic)
    Cannot-link: Pairs of articles that MUST NOT be in the same cluster (different topics)
    
    Returns:
        Tuple of (must_link_pairs, cannot_link_pairs)
        Each pair is (node_index_a, node_index_b)
    """
    random.seed(random_seed)
    
    print("Generating clustering constraints...")
    
    must_link = []
    cannot_link = []
    
    # Must-link: sample pairs within each topic
    for topic in topics:
        articles = topic['node_list']
        if len(articles) < 2:
            continue
        
        pairs = list(combinations(articles, 2))
        for art_a, art_b in pairs:
            must_link.append((art_a['node_index'], art_b['node_index']))
    
    # Cannot-link: sample pairs across topics
    for topic_a, topic_b in combinations(topics, 2):
        for art_a in topic_a['node_list']:
            for art_b in topic_b['node_list']:
                cannot_link.append((art_a['node_index'], art_b['node_index']))
    
    # Sample if too many
    if len(must_link) > sample_must_link:
        must_link = random.sample(must_link, sample_must_link)
    if len(cannot_link) > sample_cannot_link:
        cannot_link = random.sample(cannot_link, sample_cannot_link)
    
    print(f"Generated {len(must_link)} must-link constraints")
    print(f"Generated {len(cannot_link)} cannot-link constraints")
    
    return must_link, cannot_link


# =============================================================================
# STAGE 3: EVENT THREADING DATA
# =============================================================================

def classify_temporal_relationship(
    date_a: str, 
    date_b: str, 
    title_a: str, 
    title_b: str
) -> str:
    """
    Classify the temporal relationship between two articles.
    
    Categories:
    - SAME_DAY: Published on the same day (likely same event coverage)
    - IMMEDIATE_UPDATE: 1-2 days apart (follow-up coverage)
    - SHORT_TERM_DEVELOPMENT: 3-7 days apart (story development)
    - LONG_TERM_DEVELOPMENT: 8-30 days apart (ongoing story)
    - DISTANT_RELATED: 30+ days apart (same topic, distant events)
    """
    d_a = datetime.strptime(date_a, '%Y-%m-%d')
    d_b = datetime.strptime(date_b, '%Y-%m-%d')
    
    days_apart = abs((d_b - d_a).days)
    
    if days_apart == 0:
        return 'SAME_DAY'
    elif days_apart <= 2:
        return 'IMMEDIATE_UPDATE'
    elif days_apart <= 7:
        return 'SHORT_TERM_DEV'
    elif days_apart <= 30:
        return 'LONG_TERM_DEV'
    else:
        return 'DISTANT_RELATED'


def generate_event_threading_data(
    topics: List[Dict],
    max_pairs_per_topic: int = 500,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate training data for event threading classifier.
    
    This creates ordered pairs within each topic to learn:
    - Temporal relationships (how far apart)
    - Whether B is an update/development of A
    
    Args:
        topics: List of topic dictionaries
        max_pairs_per_topic: Limit pairs per topic to avoid imbalance
        random_seed: For reproducibility
        
    Returns:
        DataFrame with columns for threading classification
    """
    random.seed(random_seed)
    
    print("Generating event threading data...")
    
    threading_pairs = []
    
    for topic in topics:
        topic_idx = topic['topic_index']
        topic_name = topic.get('topic', 'Unknown')
        articles = topic['node_list']
        
        # Sort articles by date within topic
        articles_sorted = sorted(articles, key=lambda x: x['date'])
        
        pairs = []
        # Generate ordered pairs (A before B chronologically)
        for i in range(len(articles_sorted)):
            for j in range(i + 1, len(articles_sorted)):
                art_a = articles_sorted[i]
                art_b = articles_sorted[j]
                
                d_a = datetime.strptime(art_a['date'], '%Y-%m-%d')
                d_b = datetime.strptime(art_b['date'], '%Y-%m-%d')
                days_apart = (d_b - d_a).days
                
                relationship = classify_temporal_relationship(
                    art_a['date'], art_b['date'],
                    art_a['title'], art_b['title']
                )
                
                pairs.append({
                    'text_a': art_a['title'],
                    'text_b': art_b['title'],
                    'topic_index': topic_idx,
                    'topic_name': topic_name,
                    'date_a': art_a['date'],
                    'date_b': art_b['date'],
                    'days_apart': days_apart,
                    'node_a': art_a['node_index'],
                    'node_b': art_b['node_index'],
                    'temporal_relation': relationship,
                    'a_before_b': 1  # A is always chronologically before B
                })
        
        # Sample if too many pairs in this topic
        if len(pairs) > max_pairs_per_topic:
            pairs = random.sample(pairs, max_pairs_per_topic)
        
        threading_pairs.extend(pairs)
    
    df = pd.DataFrame(threading_pairs)
    
    print(f"Total threading pairs: {len(df)}")
    print(f"Temporal relation distribution:\n{df['temporal_relation'].value_counts()}")
    
    return df


# =============================================================================
# TRAIN/VAL/TEST SPLITTING
# =============================================================================

def split_topics_train_val_test(
    topics: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split topics into train/val/test sets.
    
    IMPORTANT: We split by TOPIC, not by article!
    This prevents data leakage where the model sees articles from
    the same topic in both training and testing.
    
    Returns:
        Tuple of (train_topics, val_topics, test_topics)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
    
    random.seed(random_seed)
    
    # Shuffle topics
    topics_shuffled = topics.copy()
    random.shuffle(topics_shuffled)
    
    n = len(topics_shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_topics = topics_shuffled[:train_end]
    val_topics = topics_shuffled[train_end:val_end]
    test_topics = topics_shuffled[val_end:]
    
    # Print statistics
    train_articles = sum(len(t['node_list']) for t in train_topics)
    val_articles = sum(len(t['node_list']) for t in val_topics)
    test_articles = sum(len(t['node_list']) for t in test_topics)
    
    print(f"\nTopic-level split:")
    print(f"  Train: {len(train_topics)} topics, {train_articles} articles")
    print(f"  Val:   {len(val_topics)} topics, {val_articles} articles")
    print(f"  Test:  {len(test_topics)} topics, {test_articles} articles")
    
    return train_topics, val_topics, test_topics


# =============================================================================
# ARTICLE INDEX CREATION
# =============================================================================

def create_article_index(topics: List[Dict]) -> pd.DataFrame:
    """
    Create a flat index of all articles with their topic assignments.
    
    Useful for:
    - Looking up articles by node_index
    - Mapping between article IDs and topic IDs
    - Evaluating clustering results
    """
    articles = []
    
    for topic in topics:
        for article in topic['node_list']:
            articles.append({
                'node_index': article['node_index'],
                'title': article['title'],
                'date': article['date'],
                'url': article.get('url', ''),
                'topic_index': topic['topic_index'],
                'topic_name': topic.get('topic', ''),
                'domain': topic.get('domain', '')
            })
    
    df = pd.DataFrame(articles)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_all_datasets(
    labeled_filepath: str,
    output_dir: str = './prepared_data',
    negative_ratio: float = 1.0,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Main function to prepare all datasets for the three-stage pipeline.
    
    Creates:
    1. Pairwise classification data (train/val/test)
    2. Clustering constraints
    3. Event threading data (train/val/test)
    4. Article index
    
    Args:
        labeled_filepath: Path to labeled JSON file
        output_dir: Directory to save prepared datasets
        negative_ratio: Ratio of negative to positive pairs
        random_seed: For reproducibility
        
    Returns:
        Dictionary of all generated DataFrames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    topics = load_labeled_data(labeled_filepath)
    
    # Split topics
    train_topics, val_topics, test_topics = split_topics_train_val_test(
        topics, random_seed=random_seed
    )
    
    print("\n" + "="*60)
    print("GENERATING STAGE 1: PAIRWISE CLASSIFICATION DATA")
    print("="*60)
    
    # Generate pairwise data for each split
    pairwise_train = generate_pairwise_data(
        train_topics, negative_ratio=negative_ratio, random_seed=random_seed
    )
    pairwise_val = generate_pairwise_data(
        val_topics, negative_ratio=negative_ratio, random_seed=random_seed+1
    )
    pairwise_test = generate_pairwise_data(
        test_topics, negative_ratio=negative_ratio, random_seed=random_seed+2
    )
    
    print("\n" + "="*60)
    print("GENERATING STAGE 2: CLUSTERING CONSTRAINTS")
    print("="*60)
    
    # Generate constraints from training topics
    must_link, cannot_link = generate_clustering_constraints(
        train_topics, random_seed=random_seed
    )
    
    print("\n" + "="*60)
    print("GENERATING STAGE 3: EVENT THREADING DATA")
    print("="*60)
    
    # Generate threading data
    threading_train = generate_event_threading_data(
        train_topics, random_seed=random_seed
    )
    threading_val = generate_event_threading_data(
        val_topics, random_seed=random_seed+1
    )
    threading_test = generate_event_threading_data(
        test_topics, random_seed=random_seed+2
    )
    
    print("\n" + "="*60)
    print("CREATING ARTICLE INDEX")
    print("="*60)
    
    # Create article indices
    article_index_train = create_article_index(train_topics)
    article_index_val = create_article_index(val_topics)
    article_index_test = create_article_index(test_topics)
    article_index_all = create_article_index(topics)
    
    print(f"Total articles indexed: {len(article_index_all)}")
    
    print("\n" + "="*60)
    print("SAVING DATASETS")
    print("="*60)
    
    # Save all datasets
    datasets = {
        # Stage 1: Pairwise
        'pairwise_train': pairwise_train,
        'pairwise_val': pairwise_val,
        'pairwise_test': pairwise_test,
        
        # Stage 3: Threading
        'threading_train': threading_train,
        'threading_val': threading_val,
        'threading_test': threading_test,
        
        # Article indices
        'article_index_train': article_index_train,
        'article_index_val': article_index_val,
        'article_index_test': article_index_test,
        'article_index_all': article_index_all,
    }
    
    for name, df in datasets.items():
        filepath = output_path / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved {name}: {len(df)} rows -> {filepath}")
    
    # Save constraints as JSON
    constraints = {
        'must_link': must_link,
        'cannot_link': cannot_link
    }
    constraints_path = output_path / 'clustering_constraints.json'
    with open(constraints_path, 'w') as f:
        json.dump(constraints, f, indent=2)
    print(f"Saved clustering constraints -> {constraints_path}")
    
    # Save topic splits for reference
    splits = {
        'train_topic_indices': [t['topic_index'] for t in train_topics],
        'val_topic_indices': [t['topic_index'] for t in val_topics],
        'test_topic_indices': [t['topic_index'] for t in test_topics]
    }
    splits_path = output_path / 'topic_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved topic splits -> {splits_path}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    
    return datasets


# =============================================================================
# UTILITY: CONVERT TO XGBOOST FORMAT
# =============================================================================

def convert_to_xgboost_format(
    pairwise_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Convert pairwise data to format expected by existing XGBoost code.
    
    The existing code expects:
    - text_a: First text
    - text_b: Second text  
    - label: 0 or 1
    """
    xgb_df = pairwise_df[['text_a', 'text_b', 'label']].copy()
    xgb_df.to_csv(output_path, index=False)
    print(f"Saved XGBoost format data: {len(xgb_df)} rows -> {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for timeline construction pipeline')
    parser.add_argument('--labeled', type=str, required=True,
                        help='Path to labeled data JSON file')
    parser.add_argument('--output', type=str, default='./prepared_data',
                        help='Output directory for prepared datasets')
    parser.add_argument('--negative-ratio', type=float, default=1.0,
                        help='Ratio of negative to positive pairs (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run preparation
    datasets = prepare_all_datasets(
        labeled_filepath=args.labeled,
        output_dir=args.output,
        negative_ratio=args.negative_ratio,
        random_seed=args.seed
    )
    
    # Also save in XGBoost-compatible format
    convert_to_xgboost_format(
        datasets['pairwise_train'],
        f"{args.output}/xgboost_train.csv"
    )
    convert_to_xgboost_format(
        datasets['pairwise_val'],
        f"{args.output}/xgboost_val.csv"
    )
    convert_to_xgboost_format(
        datasets['pairwise_test'],
        f"{args.output}/xgboost_test.csv"
    )
    
    print("\nAll datasets prepared successfully!")
    print(f"\nNext steps:")
    print(f"  1. Train XGBoost on: {args.output}/xgboost_train.csv")
    print(f"  2. Use constraints from: {args.output}/clustering_constraints.json")
    print(f"  3. Train threading model on: {args.output}/threading_train.csv")