# News Timeline Construction Pipeline

A three-stage pipeline for constructing news event timelines:

1. **Stage 1**: Pairwise Relatedness Classification (XGBoost)
2. **Stage 2**: Event Clustering (Leiden Algorithm)
3. **Stage 3**: Temporal Ordering (Event Threading)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: News Articles                        │
│                    (title, date, url)                               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Pairwise Relatedness (XGBoost)                           │
│  ─────────────────────────────────────────                         │
│  Input:  Two article titles                                         │
│  Output: P(related) ∈ [0, 1]                                       │
│  Features: TF-IDF + Similarity metrics                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Event Clustering (Leiden Algorithm)                       │
│  ─────────────────────────────────────────────                      │
│  Input:  Pairwise similarity scores                                 │
│  Output: Cluster assignments                                        │
│  Method: Temporal windowing → Similarity graph → Community detection│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Temporal Ordering (Event Threading)                       │
│  ─────────────────────────────────────────────                      │
│  Input:  Articles within a cluster                                  │
│  Output: Ordered timeline with relationship labels                  │
│  Classes: SAME_DAY, IMMEDIATE_UPDATE, SHORT_TERM_DEV, etc.         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Timelines                           │
│           [{position, title, date, role}, ...]                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

Note: Leiden algorithm requires:
```bash
pip install leidenalg python-igraph
```

## Quick Start

### 1. Prepare Your Data

First, run data preparation on your labeled ETimeline JSON:

```bash
python data_preparation.py --labeled your_labeled_data.json --output ./prepared_data
```

This creates:
- `xgboost_train/val/test.csv` - For Stage 1
- `threading_train/val/test.csv` - For Stage 3
- `clustering_constraints.json` - For Stage 2
- `article_index_*.csv` - Article metadata

### 2. Train the Pipeline

```bash
python main_pipeline.py --mode train --data_dir ./prepared_data
```

This trains all three stages and saves models to `./models/`.

### 3. Run Inference

```bash
python main_pipeline.py --mode inference --input new_articles.json --output timelines.json
```

### 4. Evaluate

```bash
python main_pipeline.py --mode evaluate --data_dir ./prepared_data
```

## File Structure

```
pipeline/
├── data_preparation.py      # Convert labeled data to training format
├── stage1_pairwise.py       # XGBoost pairwise classifier
├── stage2_clustering.py     # Leiden event clustering
├── stage3_threading.py      # Event threading classifier
├── main_pipeline.py         # Main entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Training Each Stage Independently

### Stage 1: Pairwise Classifier

```python
from stage1_pairwise import train_stage1

model, vectorizer, metrics = train_stage1(
    data_dir='./prepared_data',
    model_output='./models/stage1_xgboost.pkl'
)
```

### Stage 2: Clustering

```python
from stage2_clustering import EventClusterer, ClusteringConfig

config = ClusteringConfig(
    temporal_window_days=30,
    similarity_threshold=0.7,
    leiden_resolution=1.0
)

clusterer = EventClusterer(
    stage1_model_path='./models/stage1_xgboost.pkl',
    config=config
)

clusters = clusterer.fit_predict(articles_df)
```

### Stage 3: Event Threading

```python
from stage3_threading import train_stage3

classifier, metrics = train_stage3(
    data_dir='./prepared_data',
    model_output='./models/stage3_threading.pkl'
)
```

## Configuration Options

### Clustering Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temporal_window_days` | 30 | Only compare articles within N days |
| `similarity_threshold` | 0.7 | Min probability to create edge |
| `temporal_decay_lambda` | 0.05 | Decay rate for temporal weighting |
| `min_cluster_size` | 2 | Minimum articles per cluster |
| `leiden_resolution` | 1.0 | Higher = more clusters |

### Command Line Options

```bash
python main_pipeline.py --mode train \
    --data_dir ./prepared_data \
    --model_dir ./models \
    --results_dir ./results \
    --temporal_window 30 \
    --similarity_threshold 0.7 \
    --leiden_resolution 1.0
```

## Evaluation Metrics

### Stage 1 (Pairwise)
- Accuracy, Precision, Recall, F1
- ROC-AUC

### Stage 2 (Clustering)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- BCubed Precision, Recall, F1

### Stage 3 (Threading)
- Classification accuracy per relation type
- Macro/Weighted F1
- Kendall's Tau for timeline ordering

## Expected Results

Based on the dataset characteristics:

| Stage | Metric | Expected Range |
|-------|--------|----------------|
| Stage 1 | Accuracy | 90-95% |
| Stage 2 | BCubed F1 | 70-85% |
| Stage 3 | Accuracy | 60-75% |

## Troubleshooting

### "leidenalg not installed"
```bash
pip install leidenalg python-igraph
```

### Out of Memory on Stage 2
Reduce `temporal_window_days` or increase `similarity_threshold` to create fewer edges.

### Poor Clustering Results
- Try adjusting `leiden_resolution` (higher = more clusters)
- Lower `similarity_threshold` if clusters are too fragmented
- Increase `min_cluster_size` to filter noise

## References

- Leiden Algorithm: Traag, V.A., Waltman, L. & van Eck, N.J. (2019)
- Story Forest: Liu et al., ACM TKDD 2020
- Event Threading: Nallapati et al., CIKM 2004