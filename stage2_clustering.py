"""
Stage 2: Leiden Clustering for Event Detection

This stage takes pairwise similarity scores from Stage 1 (XGBoost)
and clusters articles into event groups using the Leiden algorithm.

Key features:
1. Temporal windowing - only compare articles within N days
2. Similarity thresholding - only create edges for confident predictions
3. Temporal decay - closer articles weighted higher
4. Leiden community detection - finds event clusters
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json
from itertools import combinations
import warnings

# Graph libraries
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn("leidenalg not installed. Run: pip install leidenalg python-igraph")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Import Stage 1 classifier
from stage1_pairwise import PairwiseClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

class ClusteringConfig:
    """Configuration for clustering pipeline."""
    
    def __init__(
        self,
        temporal_window_days: int = 30,
        similarity_threshold: float = 0.7,
        temporal_decay_lambda: float = 0.05,
        min_cluster_size: int = 2,
        leiden_resolution: float = 1.0,
        batch_size: int = 1000
    ):
        """
        Args:
            temporal_window_days: Only compare articles within this many days
            similarity_threshold: Minimum probability to create an edge
            temporal_decay_lambda: Decay rate for temporal weighting
            min_cluster_size: Minimum articles to form a cluster
            leiden_resolution: Leiden algorithm resolution parameter
            batch_size: Batch size for pairwise predictions
        """
        self.temporal_window_days = temporal_window_days
        self.similarity_threshold = similarity_threshold
        self.temporal_decay_lambda = temporal_decay_lambda
        self.min_cluster_size = min_cluster_size
        self.leiden_resolution = leiden_resolution
        self.batch_size = batch_size


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def get_candidate_pairs(
    articles: pd.DataFrame,
    temporal_window_days: int = 30
) -> List[Tuple[int, int]]:
    """
    Generate candidate pairs within temporal window.
    
    Instead of comparing all O(n²) pairs, only compare articles
    published within temporal_window_days of each other.
    
    Args:
        articles: DataFrame with 'date' column
        temporal_window_days: Maximum days apart to consider
        
    Returns:
        List of (idx_a, idx_b) tuples
    """
    print(f"Generating candidate pairs (window={temporal_window_days} days)...")
    
    # Ensure date is datetime
    articles = articles.copy()
    articles['date'] = pd.to_datetime(articles['date'])
    
    # Sort by date for efficient windowing
    articles_sorted = articles.sort_values('date').reset_index(drop=True)
    
    candidates = []
    n = len(articles_sorted)
    
    for i in range(n):
        date_i = articles_sorted.loc[i, 'date']
        
        # Look forward within window
        for j in range(i + 1, n):
            date_j = articles_sorted.loc[j, 'date']
            days_apart = (date_j - date_i).days
            
            if days_apart > temporal_window_days:
                break  # No need to look further (sorted by date)
            
            # Get original indices
            orig_i = articles_sorted.loc[i, 'original_idx'] if 'original_idx' in articles_sorted.columns else i
            orig_j = articles_sorted.loc[j, 'original_idx'] if 'original_idx' in articles_sorted.columns else j
            
            candidates.append((i, j))
    
    print(f"Generated {len(candidates)} candidate pairs from {n} articles")
    print(f"Reduction: {len(candidates) / (n * (n-1) / 2) * 100:.1f}% of all possible pairs")
    
    return candidates


def compute_temporal_decay(days_apart: int, lambda_decay: float = 0.05) -> float:
    """
    Compute temporal decay weight.
    
    weight = exp(-lambda * days_apart)
    
    This makes closer articles have higher edge weights.
    """
    return np.exp(-lambda_decay * days_apart)


def build_similarity_graph(
    articles: pd.DataFrame,
    classifier: PairwiseClassifier,
    config: ClusteringConfig
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Build similarity graph using pairwise classifier.
    
    Args:
        articles: DataFrame with 'title' and 'date' columns
        classifier: Trained Stage 1 classifier
        config: Clustering configuration
        
    Returns:
        (edges, weights) where edges are (i, j) tuples and weights are floats
    """
    print("\n" + "="*60)
    print("BUILDING SIMILARITY GRAPH")
    print("="*60)
    
    # Get candidate pairs
    candidates = get_candidate_pairs(articles, config.temporal_window_days)
    
    if len(candidates) == 0:
        print("Warning: No candidate pairs generated!")
        return [], []
    
    # Prepare texts for batch prediction
    texts_a = [articles.iloc[i]['title'] for i, j in candidates]
    texts_b = [articles.iloc[j]['title'] for i, j in candidates]
    
    # Get pairwise predictions
    print(f"Running pairwise predictions on {len(candidates)} pairs...")
    _, probabilities = classifier.predict_pairs_batch(
        texts_a, texts_b, 
        batch_size=config.batch_size
    )
    
    # Build edges with weights
    edges = []
    weights = []
    
    articles['date'] = pd.to_datetime(articles['date'])
    
    for idx, (i, j) in enumerate(candidates):
        prob = probabilities[idx]
        
        if prob >= config.similarity_threshold:
            # Compute temporal decay
            days_apart = abs((articles.iloc[j]['date'] - articles.iloc[i]['date']).days)
            temporal_weight = compute_temporal_decay(days_apart, config.temporal_decay_lambda)
            
            # Final edge weight = similarity * temporal_decay
            weight = prob * temporal_weight
            
            edges.append((i, j))
            weights.append(weight)
    
    print(f"Created {len(edges)} edges (threshold={config.similarity_threshold})")
    print(f"Edge density: {len(edges) / len(candidates) * 100:.1f}% of candidates")
    
    return edges, weights


# =============================================================================
# LEIDEN CLUSTERING
# =============================================================================

def leiden_clustering(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    weights: List[float],
    resolution: float = 1.0
) -> List[int]:
    """
    Run Leiden community detection algorithm.
    
    Args:
        n_nodes: Number of nodes (articles)
        edges: List of (i, j) edge tuples
        weights: Edge weights
        resolution: Resolution parameter (higher = more clusters)
        
    Returns:
        List of cluster assignments (one per node)
    """
    if not LEIDEN_AVAILABLE:
        raise ImportError("leidenalg not installed. Run: pip install leidenalg python-igraph")
    
    print(f"\nRunning Leiden clustering (resolution={resolution})...")
    
    # Create igraph Graph
    g = ig.Graph()
    g.add_vertices(n_nodes)
    g.add_edges(edges)
    g.es['weight'] = weights
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    
    clusters = partition.membership
    
    n_clusters = len(set(clusters))
    print(f"Found {n_clusters} clusters")
    
    # Cluster size distribution
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[c] += 1
    
    sizes = list(cluster_sizes.values())
    print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f}")
    
    return clusters


def fallback_connected_components(
    n_nodes: int,
    edges: List[Tuple[int, int]]
) -> List[int]:
    """
    Fallback clustering using connected components.
    
    Use this if Leiden is not available.
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("Neither leidenalg nor networkx available!")
    
    print("\nUsing fallback: Connected Components clustering...")
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)
    
    clusters = [-1] * n_nodes
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            clusters[node] = cluster_id
    
    n_clusters = len(set(clusters))
    print(f"Found {n_clusters} connected components")
    
    return clusters


# =============================================================================
# POST-PROCESSING
# =============================================================================

def filter_small_clusters(
    clusters: List[int],
    min_size: int = 2
) -> List[int]:
    """
    Filter out clusters smaller than min_size.
    
    Small clusters get assigned to cluster -1 (noise).
    """
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[c] += 1
    
    filtered = []
    for c in clusters:
        if cluster_sizes[c] >= min_size:
            filtered.append(c)
        else:
            filtered.append(-1)  # Noise cluster
    
    valid_clusters = len([c for c in set(filtered) if c != -1])
    noise_count = filtered.count(-1)
    
    print(f"After filtering (min_size={min_size}): {valid_clusters} clusters, {noise_count} noise articles")
    
    return filtered


def split_temporal_gaps(
    articles: pd.DataFrame,
    clusters: List[int],
    max_gap_days: int = 30
) -> List[int]:
    """
    Split clusters with large temporal gaps.
    
    If articles in a cluster are more than max_gap_days apart
    with no articles in between, split into separate clusters.
    """
    print(f"Checking for temporal gaps > {max_gap_days} days...")
    
    articles = articles.copy()
    articles['date'] = pd.to_datetime(articles['date'])
    articles['cluster'] = clusters
    
    new_clusters = clusters.copy()
    next_cluster_id = max(clusters) + 1
    splits = 0
    
    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue
            
        cluster_articles = articles[articles['cluster'] == cluster_id].sort_values('date')
        
        if len(cluster_articles) < 2:
            continue
        
        # Check for gaps
        dates = cluster_articles['date'].values
        indices = cluster_articles.index.tolist()
        
        current_group_start = 0
        
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).astype('timedelta64[D]').astype(int)
            
            if gap > max_gap_days:
                # Split: assign new cluster ID to articles from current_group_start to i-1
                if current_group_start > 0:  # Don't reassign the first group
                    for idx in indices[current_group_start:i]:
                        new_clusters[idx] = next_cluster_id
                    next_cluster_id += 1
                    splits += 1
                
                current_group_start = i
    
    if splits > 0:
        print(f"Split {splits} clusters due to temporal gaps")
    
    return new_clusters


# =============================================================================
# MAIN CLUSTERING PIPELINE
# =============================================================================

class EventClusterer:
    """
    Main class for Stage 2 event clustering.
    
    Usage:
        clusterer = EventClusterer(stage1_model_path='./models/stage1_xgboost.pkl')
        clusters = clusterer.fit_predict(articles_df)
    """
    
    def __init__(
        self,
        stage1_model_path: str,
        config: ClusteringConfig = None
    ):
        """
        Initialize clusterer.
        
        Args:
            stage1_model_path: Path to trained Stage 1 model
            config: Clustering configuration
        """
        self.classifier = PairwiseClassifier(stage1_model_path)
        self.config = config or ClusteringConfig()
        
        self.edges_ = None
        self.weights_ = None
        self.clusters_ = None
    
    def fit_predict(self, articles: pd.DataFrame) -> np.ndarray:
        """
        Cluster articles into events.
        
        Args:
            articles: DataFrame with 'title' and 'date' columns
            
        Returns:
            Array of cluster assignments
        """
        print("\n" + "="*60)
        print("STAGE 2: EVENT CLUSTERING")
        print("="*60)
        print(f"Clustering {len(articles)} articles...")
        
        # Build similarity graph
        self.edges_, self.weights_ = build_similarity_graph(
            articles, 
            self.classifier,
            self.config
        )
        
        if len(self.edges_) == 0:
            print("Warning: No edges created. All articles in separate clusters.")
            return np.arange(len(articles))
        
        # Run Leiden clustering
        if LEIDEN_AVAILABLE:
            clusters = leiden_clustering(
                n_nodes=len(articles),
                edges=self.edges_,
                weights=self.weights_,
                resolution=self.config.leiden_resolution
            )
        else:
            clusters = fallback_connected_components(
                n_nodes=len(articles),
                edges=self.edges_
            )
        
        # Post-processing
        clusters = filter_small_clusters(clusters, self.config.min_cluster_size)
        clusters = split_temporal_gaps(articles, clusters, self.config.temporal_window_days)
        
        self.clusters_ = np.array(clusters)
        
        # Print summary
        self._print_summary(articles)
        
        return self.clusters_
    
    def _print_summary(self, articles: pd.DataFrame):
        """Print clustering summary."""
        print("\n" + "-"*40)
        print("CLUSTERING SUMMARY")
        print("-"*40)
        
        valid_clusters = [c for c in self.clusters_ if c != -1]
        n_clusters = len(set(valid_clusters))
        n_noise = (self.clusters_ == -1).sum()
        
        print(f"Total articles: {len(articles)}")
        print(f"Clusters found: {n_clusters}")
        print(f"Noise articles: {n_noise}")
        
        if n_clusters > 0:
            cluster_sizes = pd.Series(valid_clusters).value_counts()
            print(f"Largest cluster: {cluster_sizes.max()} articles")
            print(f"Smallest cluster: {cluster_sizes.min()} articles")
            print(f"Average cluster size: {cluster_sizes.mean():.1f} articles")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_clustering(
    predicted_clusters: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate clustering quality against ground truth.
    
    Metrics:
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - BCubed Precision, Recall, F1
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Filter out noise (-1) for fair comparison
    mask = predicted_clusters != -1
    pred_filtered = predicted_clusters[mask]
    true_filtered = true_labels[mask]
    
    metrics = {}
    
    # ARI
    metrics['ari'] = adjusted_rand_score(true_filtered, pred_filtered)
    
    # NMI
    metrics['nmi'] = normalized_mutual_info_score(true_filtered, pred_filtered)
    
    # BCubed metrics
    bcubed = compute_bcubed(pred_filtered, true_filtered)
    metrics.update(bcubed)
    
    print("\n" + "="*60)
    print("CLUSTERING EVALUATION")
    print("="*60)
    print(f"Adjusted Rand Index (ARI): {metrics['ari']:.4f}")
    print(f"Normalized Mutual Info (NMI): {metrics['nmi']:.4f}")
    print(f"BCubed Precision: {metrics['bcubed_precision']:.4f}")
    print(f"BCubed Recall: {metrics['bcubed_recall']:.4f}")
    print(f"BCubed F1: {metrics['bcubed_f1']:.4f}")
    
    return metrics


def compute_bcubed(predicted: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Compute BCubed precision, recall, and F1.
    
    BCubed evaluates clustering at the item level.
    """
    n = len(predicted)
    
    # Build lookup dictionaries
    pred_clusters = defaultdict(set)
    true_clusters = defaultdict(set)
    
    for i, (p, t) in enumerate(zip(predicted, true)):
        pred_clusters[p].add(i)
        true_clusters[t].add(i)
    
    precision_sum = 0
    recall_sum = 0
    
    for i in range(n):
        pred_cluster = pred_clusters[predicted[i]]
        true_cluster = true_clusters[true[i]]
        
        # Items that share both predicted and true cluster with i
        intersection = pred_cluster & true_cluster
        
        precision_sum += len(intersection) / len(pred_cluster)
        recall_sum += len(intersection) / len(true_cluster)
    
    precision = precision_sum / n
    recall = recall_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'bcubed_precision': precision,
        'bcubed_recall': recall,
        'bcubed_f1': f1
    }


# =============================================================================
# MAIN TRAINING/EVALUATION PIPELINE
# =============================================================================

def run_stage2(
    data_dir: str = './prepared_data',
    model_dir: str = './models',
    output_dir: str = './results',
    config: ClusteringConfig = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run Stage 2 clustering pipeline.
    
    Args:
        data_dir: Directory with prepared data
        model_dir: Directory with Stage 1 model
        output_dir: Where to save results
        config: Clustering configuration
        
    Returns:
        (cluster_assignments, evaluation_metrics)
    """
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = config or ClusteringConfig()
    
    # Load test articles (we evaluate on test set)
    print("Loading test articles...")
    articles = pd.read_csv(data_path / 'article_index_test.csv')
    print(f"Loaded {len(articles)} test articles")
    
    # Initialize clusterer
    clusterer = EventClusterer(
        stage1_model_path=str(model_path / 'stage1_xgboost.pkl'),
        config=config
    )
    
    # Run clustering
    predicted_clusters = clusterer.fit_predict(articles)
    
    # Evaluate against ground truth
    true_labels = articles['topic_index'].values
    metrics = evaluate_clustering(predicted_clusters, true_labels)
    
    # Save results
    results_df = articles.copy()
    results_df['predicted_cluster'] = predicted_clusters
    results_df.to_csv(output_path / 'stage2_clustering_results.csv', index=False)
    
    with open(output_path / 'stage2_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return predicted_clusters, metrics


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # You may need to install: pip install leidenalg python-igraph
    
    config = ClusteringConfig(
        temporal_window_days=30,
        similarity_threshold=0.7,
        temporal_decay_lambda=0.05,
        min_cluster_size=2,
        leiden_resolution=1.0
    )
    
    clusters, metrics = run_stage2(
        data_dir='./prepared_data',
        model_dir='./models',
        output_dir='./results',
        config=config
    )