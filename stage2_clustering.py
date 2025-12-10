"""
Stage 2: Leiden Clustering for Event Detection
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


class ClusteringConfig:
    """Clustering configuration."""
    
    def __init__(
        self,
        temporal_window_days: int = 30,
        similarity_threshold: float = 0.6,
        temporal_decay_lambda: float = 0.05,
        min_cluster_size: int = 2,
        leiden_resolution: float = 1.0,
        batch_size: int = 1000
    ):
        self.temporal_window_days = temporal_window_days
        self.similarity_threshold = similarity_threshold
        self.temporal_decay_lambda = temporal_decay_lambda
        self.min_cluster_size = min_cluster_size
        self.leiden_resolution = leiden_resolution
        self.batch_size = batch_size


def get_candidate_pairs(
    articles: pd.DataFrame,
    temporal_window_days: int = 30
) -> List[Tuple[int, int]]:
    """Generate candidate pairs within temporal window."""
    articles = articles.copy()
    articles['date'] = pd.to_datetime(articles['date'])
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
                break
            
            orig_i = articles_sorted.loc[i, 'original_idx'] if 'original_idx' in articles_sorted.columns else i
            orig_j = articles_sorted.loc[j, 'original_idx'] if 'original_idx' in articles_sorted.columns else j
            
            candidates.append((i, j))
    
    print(f"Candidate pairs: {len(candidates)} from {n} articles")
    
    return candidates


def compute_temporal_decay(days_apart: int, lambda_decay: float = 0.05) -> float:
    """Compute temporal decay weight."""
    return np.exp(-lambda_decay * days_apart)


def build_similarity_graph(
    articles: pd.DataFrame,
    classifier: PairwiseClassifier,
    config: ClusteringConfig
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Build similarity graph."""
    candidates = get_candidate_pairs(articles, config.temporal_window_days)
    
    if len(candidates) == 0:
        return [], []
    
    texts_a = [articles.iloc[i]['title'] for i, j in candidates]
    texts_b = [articles.iloc[j]['title'] for i, j in candidates]
    
    _, probabilities = classifier.predict_pairs_batch(
        texts_a, texts_b, 
        batch_size=config.batch_size
    )
    
    edges = []
    weights = []
    articles['date'] = pd.to_datetime(articles['date'])
    
    for idx, (i, j) in enumerate(candidates):
        prob = probabilities[idx]
        
        if prob >= config.similarity_threshold:
            days_apart = abs((articles.iloc[j]['date'] - articles.iloc[i]['date']).days)
            temporal_weight = compute_temporal_decay(days_apart, config.temporal_decay_lambda)
            weight = prob * temporal_weight
            
            edges.append((i, j))
            weights.append(weight)
    
    print(f"Edges created: {len(edges)} (threshold={config.similarity_threshold})")
    
    return edges, weights


def leiden_clustering(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    weights: List[float],
    resolution: float = 1.0
) -> List[int]:
    """Run Leiden clustering."""
    if not LEIDEN_AVAILABLE:
        raise ImportError("leidenalg not installed. Run: pip install leidenalg python-igraph")
    
    g = ig.Graph()
    g.add_vertices(n_nodes)
    g.add_edges(edges)
    g.es['weight'] = weights
    
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    
    clusters = partition.membership
    n_clusters = len(set(clusters))
    
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[c] += 1
    
    sizes = list(cluster_sizes.values())
    print(f"Clusters: {n_clusters} (min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f})")
    
    return clusters


def fallback_connected_components(
    n_nodes: int,
    edges: List[Tuple[int, int]]
) -> List[int]:
    """Fallback clustering using connected components."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("Neither leidenalg nor networkx available!")
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)
    
    clusters = [-1] * n_nodes
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            clusters[node] = cluster_id
    
    n_clusters = len(set(clusters))
    print(f"Connected components: {n_clusters}")
    
    return clusters


def filter_small_clusters(
    clusters: List[int],
    min_size: int = 2
) -> List[int]:
    """Filter small clusters."""
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
    
    print(f"After filtering: {valid_clusters} clusters, {noise_count} noise")
    
    return filtered


def split_temporal_gaps(
    articles: pd.DataFrame,
    clusters: List[int],
    max_gap_days: int = 30
) -> List[int]:
    """Split clusters with large temporal gaps."""
    
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


class EventClusterer:
    """Event clustering."""
    
    def __init__(
        self,
        stage1_model_path: str,
        config: ClusteringConfig = None
    ):
        """Initialize clusterer."""
        self.classifier = PairwiseClassifier(stage1_model_path)
        self.config = config or ClusteringConfig()
        
        self.edges_ = None
        self.weights_ = None
        self.clusters_ = None
    
    def fit_predict(self, articles: pd.DataFrame) -> np.ndarray:
        """Cluster articles into events."""
        print(f"Clustering {len(articles)} articles...")
        
        self.edges_, self.weights_ = build_similarity_graph(
            articles, 
            self.classifier,
            self.config
        )
        
        if len(self.edges_) == 0:
            return np.arange(len(articles))
        
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
        
        clusters = filter_small_clusters(clusters, self.config.min_cluster_size)
        clusters = split_temporal_gaps(articles, clusters, self.config.temporal_window_days)
        
        self.clusters_ = np.array(clusters)
        self._print_summary(articles)
        
        return self.clusters_
    
    def _print_summary(self, articles: pd.DataFrame):
        """Print clustering summary."""
        valid_clusters = [c for c in self.clusters_ if c != -1]
        n_clusters = len(set(valid_clusters))
        n_noise = (self.clusters_ == -1).sum()
        
        print(f"Clusters: {n_clusters}, Noise: {n_noise}")
        
        if n_clusters > 0:
            cluster_sizes = pd.Series(valid_clusters).value_counts()
            print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, avg={cluster_sizes.mean():.1f}")


def evaluate_clustering(
    predicted_clusters: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate clustering quality."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    mask = predicted_clusters != -1
    pred_filtered = predicted_clusters[mask]
    true_filtered = true_labels[mask]
    
    metrics = {}
    metrics['ari'] = adjusted_rand_score(true_filtered, pred_filtered)
    metrics['nmi'] = normalized_mutual_info_score(true_filtered, pred_filtered)
    
    bcubed = compute_bcubed(pred_filtered, true_filtered)
    metrics.update(bcubed)
    
    print(f"ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}")
    print(f"BCubed - Precision: {metrics['bcubed_precision']:.4f}, Recall: {metrics['bcubed_recall']:.4f}, F1: {metrics['bcubed_f1']:.4f}")
    
    return metrics


def compute_bcubed(predicted: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute BCubed metrics."""
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


def run_stage2(
    data_dir: str = './prepared_data',
    model_dir: str = './models',
    output_dir: str = './results',
    config: ClusteringConfig = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run Stage 2 clustering pipeline."""
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = config or ClusteringConfig()
    
    articles = pd.read_csv(data_path / 'article_index_test.csv')
    print(f"Loaded {len(articles)} test articles")
    
    clusterer = EventClusterer(
        stage1_model_path=str(model_path / 'stage1_xgboost.pkl'),
        config=config
    )
    
    predicted_clusters = clusterer.fit_predict(articles)
    true_labels = articles['topic_index'].values
    metrics = evaluate_clustering(predicted_clusters, true_labels)
    
    results_df = articles.copy()
    results_df['predicted_cluster'] = predicted_clusters
    results_df.to_csv(output_path / 'stage2_clustering_results.csv', index=False)
    
    with open(output_path / 'stage2_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return predicted_clusters, metrics


if __name__ == "__main__":
    
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