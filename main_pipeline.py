"""
Complete Timeline Construction Pipeline

This is the main entry point that runs all three stages:
    Stage 1: Pairwise Relatedness (XGBoost)
    Stage 2: Event Clustering (Leiden)
    Stage 3: Temporal Ordering (Event Threading)

Usage:
    # Train all models
    python main_pipeline.py --mode train --data_dir ./prepared_data
    
    # Run inference on new articles
    python main_pipeline.py --mode inference --input articles.json --output timelines.json
    
    # Evaluate on test set
    python main_pipeline.py --mode evaluate --data_dir ./prepared_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import pipeline stages
from stage1_pairwise import train_stage1, PairwiseClassifier
from stage2_clustering import EventClusterer, ClusteringConfig, evaluate_clustering
from stage3_threading import (
    train_stage3, 
    EventThreadingClassifier, 
    TimelineBuilder,
    evaluate_timeline_ordering
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Configuration for the full pipeline."""
    
    def __init__(
        self,
        data_dir: str = './prepared_data',
        model_dir: str = './models',
        results_dir: str = './results',
        # Stage 1 config
        max_tfidf_features: int = 5000,
        # Stage 2 config
        temporal_window_days: int = 30,
        similarity_threshold: float = 0.7,
        temporal_decay_lambda: float = 0.05,
        min_cluster_size: int = 2,
        leiden_resolution: float = 1.0,
        # Stage 3 config
        threading_model_type: str = 'xgboost'
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        self.max_tfidf_features = max_tfidf_features
        
        self.clustering_config = ClusteringConfig(
            temporal_window_days=temporal_window_days,
            similarity_threshold=similarity_threshold,
            temporal_decay_lambda=temporal_decay_lambda,
            min_cluster_size=min_cluster_size,
            leiden_resolution=leiden_resolution
        )
        
        self.threading_model_type = threading_model_type
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class TimelinePipeline:
    """
    Complete pipeline for news timeline construction.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        self.pairwise_classifier = None
        self.event_clusterer = None
        self.threading_classifier = None
        self.timeline_builder = None
        
        self.metrics = {}
    
    def train(self, skip_stage1: bool = False, skip_stage2: bool = False, skip_stage3: bool = False):
        """
        Train all pipeline components.
        
        Args:
            skip_stage1: Skip training Stage 1 (use existing model)
            skip_stage2: Skip Stage 2 evaluation (it's unsupervised)
            skip_stage3: Skip training Stage 3 (use existing model)
        """
        print("\n" + "="*70)
        print("TIMELINE CONSTRUCTION PIPELINE - TRAINING")
        print("="*70)
        
        # Stage 1: Pairwise Classifier
        if not skip_stage1:
            print("\n" + "="*70)
            print("STAGE 1: Training Pairwise Relatedness Classifier")
            print("="*70)
            
            model, vectorizer, metrics = train_stage1(
                data_dir=str(self.config.data_dir),
                model_output=str(self.config.model_dir / 'stage1_xgboost.pkl'),
                max_tfidf_features=self.config.max_tfidf_features
            )
            self.metrics['stage1'] = metrics
        
        # Load Stage 1 model
        self.pairwise_classifier = PairwiseClassifier(
            str(self.config.model_dir / 'stage1_xgboost.pkl')
        )
        
        # Stage 2: Clustering (no training, but we can evaluate)
        if not skip_stage2:
            print("\n" + "="*70)
            print("STAGE 2: Evaluating Event Clustering")
            print("="*70)
            
            self.event_clusterer = EventClusterer(
                stage1_model_path=str(self.config.model_dir / 'stage1_xgboost.pkl'),
                config=self.config.clustering_config
            )
            
            # Run on test set
            test_articles = pd.read_csv(self.config.data_dir / 'article_index_test.csv')
            predicted_clusters = self.event_clusterer.fit_predict(test_articles)
            
            true_labels = test_articles['topic_index'].values
            clustering_metrics = evaluate_clustering(predicted_clusters, true_labels)
            self.metrics['stage2'] = clustering_metrics
            
            # Save clustering results
            test_articles['predicted_cluster'] = predicted_clusters
            test_articles.to_csv(
                self.config.results_dir / 'stage2_test_clusters.csv',
                index=False
            )
        
        # Stage 3: Threading Classifier
        if not skip_stage3:
            print("\n" + "="*70)
            print("STAGE 3: Training Event Threading Classifier")
            print("="*70)
            
            classifier, metrics = train_stage3(
                data_dir=str(self.config.data_dir),
                model_output=str(self.config.model_dir / 'stage3_threading.pkl')
            )
            self.metrics['stage3'] = metrics
        
        # Load Stage 3 model
        self.threading_classifier = EventThreadingClassifier.load(
            str(self.config.model_dir / 'stage3_threading.pkl')
        )
        self.timeline_builder = TimelineBuilder(self.threading_classifier)
        
        # Save all metrics
        self._save_metrics()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        self._print_summary()
    
    def load_models(self):
        """Load pre-trained models."""
        print("Loading pre-trained models...")
        
        self.pairwise_classifier = PairwiseClassifier(
            str(self.config.model_dir / 'stage1_xgboost.pkl')
        )
        
        self.event_clusterer = EventClusterer(
            stage1_model_path=str(self.config.model_dir / 'stage1_xgboost.pkl'),
            config=self.config.clustering_config
        )
        
        self.threading_classifier = EventThreadingClassifier.load(
            str(self.config.model_dir / 'stage3_threading.pkl')
        )
        
        self.timeline_builder = TimelineBuilder(self.threading_classifier)
        
        print("Models loaded successfully!")
    
    def predict(self, articles: pd.DataFrame) -> Dict[int, List[Dict]]:
        """
        Run full pipeline on new articles.
        
        Args:
            articles: DataFrame with 'title' and 'date' columns
            
        Returns:
            Dictionary mapping cluster_id to timeline (list of articles in order)
        """
        print("\n" + "="*70)
        print("RUNNING INFERENCE PIPELINE")
        print("="*70)
        
        if self.pairwise_classifier is None:
            self.load_models()
        
        # Stage 2: Cluster articles
        print("\nStep 1: Clustering articles into events...")
        clusters = self.event_clusterer.fit_predict(articles)
        articles['cluster'] = clusters
        
        # Stage 3: Build timelines for each cluster
        print("\nStep 2: Building timelines for each cluster...")
        timelines = {}
        
        unique_clusters = [c for c in set(clusters) if c != -1]
        print(f"Building timelines for {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            cluster_articles = articles[articles['cluster'] == cluster_id]
            
            timeline = self.timeline_builder.build_timeline(
                cluster_articles,
                method='hybrid'
            )
            
            timelines[cluster_id] = timeline
        
        # Handle noise articles (cluster = -1)
        noise_articles = articles[articles['cluster'] == -1]
        if len(noise_articles) > 0:
            print(f"\n{len(noise_articles)} articles not assigned to any cluster (noise)")
            timelines[-1] = [
                {
                    'position': i,
                    'title': row['title'],
                    'date': str(row['date']),
                    'role': 'unclustered',
                    'node_index': row.get('node_index', None)
                }
                for i, (_, row) in enumerate(noise_articles.iterrows())
            ]
        
        print(f"\nGenerated {len(timelines)} timelines")
        
        return timelines
    
    def evaluate(self) -> Dict[str, Dict]:
        """
        Evaluate full pipeline on test set.
        
        Returns:
            Dictionary with metrics for each stage
        """
        print("\n" + "="*70)
        print("EVALUATING FULL PIPELINE")
        print("="*70)
        
        if self.pairwise_classifier is None:
            self.load_models()
        
        # Load test data
        test_articles = pd.read_csv(self.config.data_dir / 'article_index_test.csv')
        
        # Run pipeline
        timelines = self.predict(test_articles)
        
        # Evaluate clustering (Stage 2)
        true_labels = test_articles['topic_index'].values
        predicted_clusters = test_articles['cluster'].values
        
        clustering_metrics = evaluate_clustering(predicted_clusters, true_labels)
        
        # Evaluate ordering (Stage 3)
        # Build true timelines from test data
        true_timelines = {}
        for topic_id, group in test_articles.groupby('topic_index'):
            sorted_group = group.sort_values('date')
            true_timelines[topic_id] = sorted_group['node_index'].tolist()
        
        # Map predicted timelines to node indices
        predicted_timelines = {}
        for cluster_id, timeline in timelines.items():
            if cluster_id != -1:
                predicted_timelines[cluster_id] = [
                    entry['node_index'] for entry in timeline
                    if entry['node_index'] is not None
                ]
        
        # Note: ordering evaluation requires mapping between predicted clusters and true topics
        # This is a simplified version
        
        all_metrics = {
            'clustering': clustering_metrics,
            'n_articles': len(test_articles),
            'n_predicted_clusters': len([c for c in set(predicted_clusters) if c != -1]),
            'n_true_topics': len(true_timelines),
            'noise_ratio': (predicted_clusters == -1).mean()
        }
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nClustering Metrics:")
        print(f"  ARI: {clustering_metrics['ari']:.4f}")
        print(f"  NMI: {clustering_metrics['nmi']:.4f}")
        print(f"  BCubed F1: {clustering_metrics['bcubed_f1']:.4f}")
        print(f"\nCoverage:")
        print(f"  Articles: {all_metrics['n_articles']}")
        print(f"  Predicted clusters: {all_metrics['n_predicted_clusters']}")
        print(f"  True topics: {all_metrics['n_true_topics']}")
        print(f"  Noise ratio: {all_metrics['noise_ratio']:.2%}")
        
        return all_metrics
    
    def _save_metrics(self):
        """Save training metrics to file."""
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for stage, stage_metrics in self.metrics.items():
            metrics_json[stage] = {}
            for key, value in stage_metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[stage][key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    metrics_json[stage][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    metrics_json[stage][key] = int(value)
                else:
                    metrics_json[stage][key] = value
        
        with open(self.config.results_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"\nMetrics saved to {self.config.results_dir / 'training_metrics.json'}")
    
    def _print_summary(self):
        """Print training summary."""
        print("\n" + "-"*50)
        print("TRAINING SUMMARY")
        print("-"*50)
        
        if 'stage1' in self.metrics:
            print(f"\nStage 1 (Pairwise Classifier):")
            print(f"  Accuracy: {self.metrics['stage1']['accuracy']:.4f}")
            print(f"  F1-Score: {self.metrics['stage1']['f1_score']:.4f}")
        
        if 'stage2' in self.metrics:
            print(f"\nStage 2 (Clustering):")
            print(f"  ARI: {self.metrics['stage2']['ari']:.4f}")
            print(f"  BCubed F1: {self.metrics['stage2']['bcubed_f1']:.4f}")
        
        if 'stage3' in self.metrics:
            print(f"\nStage 3 (Threading):")
            print(f"  Accuracy: {self.metrics['stage3']['accuracy']:.4f}")
            print(f"  Macro F1: {self.metrics['stage3']['macro_f1']:.4f}")
        
        print(f"\nModels saved to: {self.config.model_dir}")
        print(f"Results saved to: {self.config.results_dir}")


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def load_articles_from_json(filepath: str) -> pd.DataFrame:
    """Load articles from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both list format and ETimeline format
    if isinstance(data, list):
        if len(data) > 0 and 'node_list' in data[0]:
            # ETimeline format
            articles = []
            for topic in data:
                for node in topic['node_list']:
                    articles.append({
                        'title': node['title'],
                        'date': node['date'],
                        'url': node.get('url', ''),
                        'node_index': node.get('node_index', ''),
                        'topic_index': topic.get('topic_index', '')
                    })
            return pd.DataFrame(articles)
        else:
            # Simple list format
            return pd.DataFrame(data)
    
    return pd.DataFrame(data)


def save_timelines_to_json(timelines: Dict, filepath: str):
    """Save timelines to JSON file."""
    # Convert to serializable format
    output = {}
    for cluster_id, timeline in timelines.items():
        output[str(cluster_id)] = timeline
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Timelines saved to {filepath}")


def format_timeline_report(timelines: Dict) -> str:
    """Format timelines as human-readable report."""
    lines = []
    lines.append("="*70)
    lines.append("TIMELINE REPORT")
    lines.append("="*70)
    
    for cluster_id, timeline in timelines.items():
        if cluster_id == -1:
            lines.append(f"\n--- UNCLUSTERED ARTICLES ({len(timeline)} articles) ---")
        else:
            lines.append(f"\n--- CLUSTER {cluster_id} ({len(timeline)} articles) ---")
        
        for entry in timeline:
            role_icon = {
                'initial': '🔵',
                'update': '➡️',
                'development': '📈',
                'conclusion': '🔴',
                'unclustered': '❓'
            }.get(entry['role'], '•')
            
            lines.append(f"  {role_icon} [{entry['date']}] {entry['title'][:70]}...")
    
    return '\n'.join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='News Timeline Construction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all models
    python main_pipeline.py --mode train --data_dir ./prepared_data
    
    # Run inference on new articles
    python main_pipeline.py --mode inference --input articles.json --output timelines.json
    
    # Evaluate on test set
    python main_pipeline.py --mode evaluate --data_dir ./prepared_data
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference', 'evaluate'],
                        help='Pipeline mode')
    parser.add_argument('--data_dir', type=str, default='./prepared_data',
                        help='Directory with prepared data')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory to save/load models')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--input', type=str,
                        help='Input JSON file for inference')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for timelines')
    
    # Stage-specific options
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Skip Stage 1 training')
    parser.add_argument('--skip_stage2', action='store_true',
                        help='Skip Stage 2 evaluation')
    parser.add_argument('--skip_stage3', action='store_true',
                        help='Skip Stage 3 training')
    
    # Hyperparameters
    parser.add_argument('--temporal_window', type=int, default=30,
                        help='Temporal window in days for clustering')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Similarity threshold for edge creation')
    parser.add_argument('--leiden_resolution', type=float, default=1.0,
                        help='Leiden algorithm resolution parameter')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        temporal_window_days=args.temporal_window,
        similarity_threshold=args.similarity_threshold,
        leiden_resolution=args.leiden_resolution
    )
    
    # Initialize pipeline
    pipeline = TimelinePipeline(config)
    
    if args.mode == 'train':
        pipeline.train(
            skip_stage1=args.skip_stage1,
            skip_stage2=args.skip_stage2,
            skip_stage3=args.skip_stage3
        )
    
    elif args.mode == 'inference':
        if not args.input:
            raise ValueError("--input required for inference mode")
        
        # Load articles
        articles = load_articles_from_json(args.input)
        print(f"Loaded {len(articles)} articles from {args.input}")
        
        # Run pipeline
        timelines = pipeline.predict(articles)
        
        # Save results
        output_path = args.output or (config.results_dir / 'timelines.json')
        save_timelines_to_json(timelines, str(output_path))
        
        # Print report
        print("\n" + format_timeline_report(timelines))
    
    elif args.mode == 'evaluate':
        metrics = pipeline.evaluate()
        
        # Save evaluation results
        with open(config.results_dir / 'evaluation_metrics.json', 'w') as f:
            # Convert any non-serializable types
            json.dump({
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in metrics.items()
                if not isinstance(v, dict)
            }, f, indent=2)


if __name__ == "__main__":
    main()