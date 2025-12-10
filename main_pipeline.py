"""
Timeline Construction Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import stages
from stage1_pairwise import train_stage1, PairwiseClassifier, Stage1Config
from stage2_clustering import EventClusterer, ClusteringConfig, evaluate_clustering
from stage3_threading import (
    train_stage3,
    EventThreadingClassifier,
    TimelineBuilder
)


class PipelineConfig:
    """Pipeline configuration."""
    
    def __init__(
        self,
        data_dir: str = './prepared_data',
        model_dir: str = './models',
        results_dir: str = './results',
        max_train_samples: int = 150000,
        max_test_samples: int = 50000,
        use_embeddings: bool = True,
        temporal_window_days: int = 30,
        similarity_threshold: float = 0.75,
        temporal_decay_lambda: float = 0.05,
        min_cluster_size: int = 2,
        leiden_resolution: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        self.stage1_config = Stage1Config(
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            use_sentence_embeddings=use_embeddings
        )
        
        self.clustering_config = ClusteringConfig(
            temporal_window_days=temporal_window_days,
            similarity_threshold=similarity_threshold,
            temporal_decay_lambda=temporal_decay_lambda,
            min_cluster_size=min_cluster_size,
            leiden_resolution=leiden_resolution
        )
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


class TimelinePipeline:
    """News timeline construction pipeline."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        self.pairwise_classifier = None
        self.event_clusterer = None
        self.threading_classifier = None
        self.timeline_builder = None
        
        self.metrics = {}
    
    def train(
        self,
        skip_stage1: bool = False,
        skip_stage2: bool = False,
        skip_stage3: bool = False
    ):
        """Train all pipeline stages."""
        if not skip_stage1:
            print("\nStage 1: Pairwise Classifier")
            model, vectorizer, metrics = train_stage1(
                data_dir=str(self.config.data_dir),
                model_output=str(self.config.model_dir / 'stage1_xgboost.pkl'),
                config=self.config.stage1_config
            )
            self.metrics['stage1'] = metrics
        
        self.pairwise_classifier = PairwiseClassifier(
            str(self.config.model_dir / 'stage1_xgboost.pkl')
        )
        
        if not skip_stage2:
            print("\nStage 2: Event Clustering")
            self.event_clusterer = EventClusterer(
                stage1_model_path=str(self.config.model_dir / 'stage1_xgboost.pkl'),
                config=self.config.clustering_config
            )
            
            test_articles = pd.read_csv(self.config.data_dir / 'article_index_test.csv')
            predicted_clusters = self.event_clusterer.fit_predict(test_articles)
            
            true_labels = test_articles['topic_index'].values
            clustering_metrics = evaluate_clustering(predicted_clusters, true_labels)
            self.metrics['stage2'] = clustering_metrics
            
            test_articles['predicted_cluster'] = predicted_clusters
            test_articles.to_csv(
                self.config.results_dir / 'stage2_test_clusters.csv',
                index=False
            )
        
        if not skip_stage3:
            print("\nStage 3: Event Threading")
            classifier, metrics = train_stage3(
                data_dir=str(self.config.data_dir),
                model_output=str(self.config.model_dir / 'stage3_threading.pkl')
            )
            self.metrics['stage3'] = metrics
        
        self.threading_classifier = EventThreadingClassifier.load(
            str(self.config.model_dir / 'stage3_threading.pkl')
        )
        self.timeline_builder = TimelineBuilder(self.threading_classifier)
        
        self._save_metrics()
        self._print_summary()
    
    def load_models(self):
        """Load pre-trained models."""
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
    
    def predict(self, articles: pd.DataFrame) -> Dict[int, List[Dict]]:
        """Run full pipeline on new articles."""
        if self.pairwise_classifier is None:
            self.load_models()
        
        print("\nRunning inference...")
        print("Clustering articles...")
        clusters = self.event_clusterer.fit_predict(articles)
        articles['cluster'] = clusters
        
        print("Building timelines...")
        
        timelines = {}
        unique_clusters = [c for c in set(clusters) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_articles = articles[articles['cluster'] == cluster_id]
            timeline = self.timeline_builder.build_timeline(cluster_articles, method='hybrid')
            timelines[cluster_id] = timeline
        
        noise_articles = articles[articles['cluster'] == -1]
        if len(noise_articles) > 0:
            timelines[-1] = [
                {'position': i, 'title': row['title'], 'date': str(row['date']), 'role': 'unclustered'}
                for i, (_, row) in enumerate(noise_articles.iterrows())
            ]
        
        print(f"Generated {len(timelines)} timelines")
        return timelines
    
    def _save_metrics(self):
        """Save metrics to file."""
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
    
    def _print_summary(self):
        """Print training summary."""
        if 'stage1' in self.metrics:
            acc = self.metrics['stage1']['accuracy']
            f1 = self.metrics['stage1']['f1_score']
            print(f"Stage 1 - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        if 'stage2' in self.metrics:
            ari = self.metrics['stage2']['ari']
            bcubed = self.metrics['stage2']['bcubed_f1']
            print(f"Stage 2 - ARI: {ari:.4f}, BCubed F1: {bcubed:.4f}")
        
        if 'stage3' in self.metrics:
            acc = self.metrics['stage3']['accuracy']
            f1 = self.metrics['stage3']['macro_f1']
            print(f"Stage 3 - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='News Timeline Pipeline')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--max_train', type=int, default=150000)
    parser.add_argument('--max_test', type=int, default=50000)
    parser.add_argument('--no_embeddings', action='store_true',
                        help='Disable sentence embeddings')
    parser.add_argument('--temporal_window', type=int, default=30)
    parser.add_argument('--similarity_threshold', type=float, default=0.75)
    parser.add_argument('--leiden_resolution', type=float, default=1.0)
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')
    parser.add_argument('--skip_stage3', action='store_true')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        use_embeddings=not args.no_embeddings,
        temporal_window_days=args.temporal_window,
        similarity_threshold=args.similarity_threshold,
        leiden_resolution=args.leiden_resolution
    )
    
    pipeline = TimelinePipeline(config)
    
    if args.mode == 'train':
        pipeline.train(
            skip_stage1=args.skip_stage1,
            skip_stage2=args.skip_stage2,
            skip_stage3=args.skip_stage3
        )
    
    elif args.mode == 'evaluate':
        pipeline.load_models()
        test_articles = pd.read_csv(config.data_dir / 'article_index_test.csv')
        timelines = pipeline.predict(test_articles)
        
        output_path = config.results_dir / 'timelines.json'
        with open(output_path, 'w') as f:
            json.dump({str(k): v for k, v in timelines.items()}, f, indent=2)
        print(f"Timelines saved to {output_path}")
    
    elif args.mode == 'inference':
        if not args.input:
            raise ValueError("--input required for inference mode")
        
        pipeline.load_models()
        articles = pd.read_json(args.input)
        timelines = pipeline.predict(articles)
        
        output_path = args.output or (config.results_dir / 'timelines.json')
        with open(output_path, 'w') as f:
            json.dump({str(k): v for k, v in timelines.items()}, f, indent=2)


if __name__ == "__main__":
    main()