# Running the Timeline Pipeline

1) Install deps (with embeddings + Leiden):
```bash
pip install -r requirements.txt
pip install leidenalg python-igraph sentence-transformers
```

2) Prepare data (labeled ETimeline JSON → CSVs):
```bash
python data-preperation.py --labeled ETimeline_timeline.json --output prepared_data
```

3) Train all stages (models land in `models/`, metrics in `results/`):
```bash
python main_pipeline.py --mode train --data_dir prepared_data
```

4) Evaluate on the held-out split (writes timelines to `results/timelines.json`):
```bash
python main_pipeline.py --mode evaluate --data_dir prepared_data
```

5) Inference on new articles (expects JSON array with `title` and `date`):
```bash
python main_pipeline.py --mode inference --input new_articles.json --output timelines.json
```
