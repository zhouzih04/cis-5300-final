"""
Stage 3: Event Threading for Temporal Ordering

This stage takes clustered articles from Stage 2 and determines
the temporal/narrative ordering within each cluster.

Key features:
1. Classifies relationship between article pairs (SAME_DAY, UPDATE, DEVELOPMENT, etc.)
2. Builds a directed graph of article relationships
3. Extracts timeline ordering using topological sort
4. Identifies key events vs. follow-up coverage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import pickle
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# =============================================================================
# TEMPORAL RELATIONSHIP TYPES
# =============================================================================

TEMPORAL_RELATIONS = {
    'SAME_DAY': 0,           # Same day coverage of same event
    'IMMEDIATE_UPDATE': 1,    # 1-2 days: Direct follow-up
    'SHORT_TERM_DEV': 2,      # 3-7 days: Story development
    'LONG_TERM_DEV': 3,       # 8-30 days: Ongoing story
    'DISTANT_RELATED': 4      # 30+ days: Same topic, distant events
}


# =============================================================================
# FEATURE ENGINEERING FOR THREADING
# =============================================================================

def extract_threading_features(
    title_a: str,
    title_b: str,
    date_a: str,
    date_b: str,
    tfidf_vectorizer: TfidfVectorizer = None
) -> Dict[str, float]:
    """
    Extract features for predicting temporal relationship.
    
    Features:
    - Temporal: days apart, which is earlier
    - Lexical: word overlap, shared entities, novelty
    - Semantic: TF-IDF cosine similarity
    - Linguistic: backward reference indicators
    """
    features = {}
    
    # Parse dates
    d_a = datetime.strptime(date_a, '%Y-%m-%d') if isinstance(date_a, str) else date_a
    d_b = datetime.strptime(date_b, '%Y-%m-%d') if isinstance(date_b, str) else date_b
    
    # Temporal features
    days_apart = abs((d_b - d_a).days)
    features['days_apart'] = days_apart
    features['log_days_apart'] = np.log1p(days_apart)
    features['a_is_earlier'] = 1 if d_a <= d_b else 0
    
    # Preprocess text
    text_a = str(title_a).lower().strip()
    text_b = str(title_b).lower().strip()
    
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    
    # Lexical overlap features
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    
    features['jaccard'] = intersection / union if union > 0 else 0
    features['common_words'] = intersection
    features['words_only_in_a'] = len(words_a - words_b)
    features['words_only_in_b'] = len(words_b - words_a)
    
    # Novelty: how much new content in B?
    features['novelty_ratio'] = len(words_b - words_a) / len(words_b) if len(words_b) > 0 else 0
    
    # Length features
    features['len_a'] = len(words_a)
    features['len_b'] = len(words_b)
    features['len_diff'] = abs(len(words_a) - len(words_b))
    features['len_ratio'] = min(len(words_a), len(words_b)) / max(len(words_a), len(words_b)) if max(len(words_a), len(words_b)) > 0 else 0
    
    # Character length
    features['char_len_a'] = len(text_a)
    features['char_len_b'] = len(text_b)
    
    # Backward reference indicators (B references A)
    backward_indicators = [
        'after', 'following', 'update', 'latest', 'new', 'now',
        'continues', 'ongoing', 'still', 'again', 'another',
        'response', 'reaction', 'aftermath'
    ]
    features['backward_ref_count'] = sum(1 for ind in backward_indicators if ind in text_b)
    
    # Forward reference indicators (A anticipates B)
    forward_indicators = [
        'will', 'expected', 'planned', 'upcoming', 'tomorrow',
        'next', 'future', 'soon'
    ]
    features['forward_ref_count'] = sum(1 for ind in forward_indicators if ind in text_a)
    
    # Event type indicators
    breaking_indicators = ['breaking', 'just in', 'alert', 'urgent']
    features['is_breaking_a'] = 1 if any(ind in text_a for ind in breaking_indicators) else 0
    features['is_breaking_b'] = 1 if any(ind in text_b for ind in breaking_indicators) else 0
    
    # TF-IDF cosine similarity (if vectorizer provided)
    if tfidf_vectorizer is not None:
        vec_a = tfidf_vectorizer.transform([text_a]).toarray()[0]
        vec_b = tfidf_vectorizer.transform([text_b]).toarray()[0]
        
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        features['tfidf_cosine'] = dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
    
    return features


def extract_features_batch(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer = None
) -> pd.DataFrame:
    """
    Extract features for a batch of article pairs.
    
    Args:
        df: DataFrame with text_a, text_b, date_a, date_b columns
        tfidf_vectorizer: Optional pre-fitted vectorizer
        
    Returns:
        DataFrame with feature columns
    """
    print(f"Extracting features for {len(df)} pairs...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        features = extract_threading_features(
            row['text_a'], row['text_b'],
            row['date_a'], row['date_b'],
            tfidf_vectorizer
        )
        features_list.append(features)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} pairs...")
    
    features_df = pd.DataFrame(features_list)
    
    return features_df


# =============================================================================
# EVENT THREADING CLASSIFIER
# =============================================================================

class EventThreadingClassifier:
    """
    Classifier for predicting temporal relationships between articles.
    
    Predicts: SAME_DAY, IMMEDIATE_UPDATE, SHORT_TERM_DEV, LONG_TERM_DEV, DISTANT_RELATED
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize classifier.
        
        Args:
            model_type: 'xgboost', 'rf' (random forest), or 'gb' (gradient boosting)
        """
        self.model_type = model_type
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None
    ):
        """
        Train the threading classifier.
        
        Args:
            train_df: Training data with text_a, text_b, date_a, date_b, temporal_relation
            val_df: Optional validation data for early stopping
        """
        print("\n" + "="*60)
        print("TRAINING EVENT THREADING CLASSIFIER")
        print("="*60)
        
        # Fit TF-IDF on training texts
        print("Fitting TF-IDF vectorizer...")
        all_texts = pd.concat([train_df['text_a'], train_df['text_b']]).apply(str.lower)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(all_texts)
        
        # Extract features
        X_train = extract_features_batch(train_df, self.tfidf_vectorizer)
        self.feature_columns = X_train.columns.tolist()
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(train_df['temporal_relation'])
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Initialize model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(self.label_encoder.classes_),
                max_depth=6,
                learning_rate=0.1,
                n_estimators=150,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        # Train
        print(f"Training {self.model_type} model...")
        
        if val_df is not None and self.model_type == 'xgboost':
            X_val = extract_features_batch(val_df, self.tfidf_vectorizer)
            y_val = self.label_encoder.transform(val_df['temporal_relation'])
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict temporal relations for article pairs.
        
        Args:
            df: DataFrame with text_a, text_b, date_a, date_b
            
        Returns:
            Array of predicted relation labels
        """
        X = extract_features_batch(df, self.tfidf_vectorizer)
        X = X[self.feature_columns]  # Ensure correct column order
        
        y_pred = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(y_pred)
        
        return labels
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each temporal relation.
        """
        X = extract_features_batch(df, self.tfidf_vectorizer)
        X = X[self.feature_columns]
        
        return self.model.predict_proba(X)
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate classifier on test data.
        """
        print("\n" + "="*60)
        print("THREADING CLASSIFIER EVALUATION")
        print("="*60)
        
        y_true = test_df['temporal_relation'].values
        y_pred = self.predict(test_df)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EventThreadingClassifier':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.tfidf_vectorizer = model_data['tfidf_vectorizer']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        
        return classifier


# =============================================================================
# TIMELINE CONSTRUCTION
# =============================================================================

class TimelineBuilder:
    """
    Builds ordered timelines from clustered articles using threading predictions.
    """
    
    def __init__(self, threading_classifier: EventThreadingClassifier):
        """
        Initialize timeline builder.
        
        Args:
            threading_classifier: Trained EventThreadingClassifier
        """
        self.classifier = threading_classifier
    
    def build_timeline(
        self,
        articles: pd.DataFrame,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """
        Build ordered timeline for a set of related articles.
        
        Args:
            articles: DataFrame with title, date columns (all from same cluster)
            method: 'date' (simple), 'graph' (threading-based), 'hybrid' (combined)
            
        Returns:
            List of articles in timeline order with metadata
        """
        if len(articles) == 0:
            return []
        
        if len(articles) == 1:
            return [self._article_to_dict(articles.iloc[0], position=0)]
        
        articles = articles.copy()
        articles['date'] = pd.to_datetime(articles['date'])
        
        if method == 'date':
            return self._build_by_date(articles)
        elif method == 'graph':
            return self._build_by_graph(articles)
        else:  # hybrid
            return self._build_hybrid(articles)
    
    def _article_to_dict(self, article: pd.Series, position: int, role: str = 'event') -> Dict:
        """Convert article to timeline entry."""
        return {
            'position': position,
            'title': article['title'],
            'date': str(article['date'].date()) if hasattr(article['date'], 'date') else str(article['date']),
            'role': role,  # 'initial', 'update', 'development', 'conclusion'
            'node_index': article.get('node_index', None)
        }
    
    def _build_by_date(self, articles: pd.DataFrame) -> List[Dict]:
        """Simple chronological ordering."""
        sorted_articles = articles.sort_values('date').reset_index(drop=True)
        
        timeline = []
        for i, (_, article) in enumerate(sorted_articles.iterrows()):
            role = 'initial' if i == 0 else 'update'
            timeline.append(self._article_to_dict(article, i, role))
        
        return timeline
    
    def _build_by_graph(self, articles: pd.DataFrame) -> List[Dict]:
        """
        Build timeline using threading predictions to create dependency graph.
        """
        if not NETWORKX_AVAILABLE:
            print("Warning: networkx not available, falling back to date ordering")
            return self._build_by_date(articles)
        
        # Create pairs for prediction
        pairs = []
        indices = articles.index.tolist()
        
        for i, idx_a in enumerate(indices):
            for j, idx_b in enumerate(indices):
                if i < j:
                    art_a = articles.loc[idx_a]
                    art_b = articles.loc[idx_b]
                    
                    pairs.append({
                        'text_a': art_a['title'],
                        'text_b': art_b['title'],
                        'date_a': str(art_a['date'].date()) if hasattr(art_a['date'], 'date') else str(art_a['date']),
                        'date_b': str(art_b['date'].date()) if hasattr(art_b['date'], 'date') else str(art_b['date']),
                        'idx_a': idx_a,
                        'idx_b': idx_b
                    })
        
        if len(pairs) == 0:
            return self._build_by_date(articles)
        
        pairs_df = pd.DataFrame(pairs)
        
        # Predict relationships
        relations = self.classifier.predict(pairs_df)
        
        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(indices)
        
        # Add edges based on temporal relationships
        for (_, row), relation in zip(pairs_df.iterrows(), relations):
            idx_a, idx_b = row['idx_a'], row['idx_b']
            date_a = pd.to_datetime(row['date_a'])
            date_b = pd.to_datetime(row['date_b'])
            
            # Determine direction based on dates and relationship
            if date_a <= date_b:
                earlier, later = idx_a, idx_b
            else:
                earlier, later = idx_b, idx_a
            
            # Add edge from earlier to later (dependency)
            if relation in ['IMMEDIATE_UPDATE', 'SHORT_TERM_DEV']:
                G.add_edge(earlier, later, relation=relation, weight=2)
            elif relation == 'SAME_DAY':
                G.add_edge(earlier, later, relation=relation, weight=1)
        
        # Topological sort (or approximate if cycles exist)
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, fall back to date-based with graph hints
            order = sorted(indices, key=lambda x: articles.loc[x, 'date'])
        
        # Build timeline
        timeline = []
        for i, idx in enumerate(order):
            article = articles.loc[idx]
            
            # Determine role
            if i == 0:
                role = 'initial'
            elif G.in_degree(idx) > 1:
                role = 'development'
            else:
                role = 'update'
            
            timeline.append(self._article_to_dict(article, i, role))
        
        return timeline
    
    def _build_hybrid(self, articles: pd.DataFrame) -> List[Dict]:
        """
        Hybrid approach: use dates as primary, threading to resolve ties.
        """
        # Group by date
        articles = articles.copy()
        articles['date_only'] = articles['date'].dt.date
        
        timeline = []
        position = 0
        
        for date, group in articles.groupby('date_only', sort=True):
            if len(group) == 1:
                # Single article on this date
                article = group.iloc[0]
                role = 'initial' if position == 0 else 'update'
                timeline.append(self._article_to_dict(article, position, role))
                position += 1
            else:
                # Multiple articles on same date - use threading to order
                sub_timeline = self._order_same_day(group, position)
                timeline.extend(sub_timeline)
                position += len(sub_timeline)
        
        return timeline
    
    def _order_same_day(self, articles: pd.DataFrame, start_position: int) -> List[Dict]:
        """Order articles from the same day using content analysis."""
        # Simple heuristic: breaking news first, then by title length (shorter = headline)
        articles = articles.copy()
        
        # Score articles
        def score_article(title):
            title_lower = title.lower()
            score = 0
            
            # Breaking news indicators
            if any(ind in title_lower for ind in ['breaking', 'just in', 'alert']):
                score += 10
            
            # Shorter titles often more significant
            score -= len(title.split()) * 0.1
            
            return score
        
        articles['_score'] = articles['title'].apply(score_article)
        sorted_articles = articles.sort_values('_score', ascending=False)
        
        timeline = []
        for i, (_, article) in enumerate(sorted_articles.iterrows()):
            role = 'initial' if start_position + i == 0 else 'update'
            timeline.append(self._article_to_dict(article, start_position + i, role))
        
        return timeline


# =============================================================================
# EVALUATION METRICS FOR ORDERING
# =============================================================================

def kendall_tau(predicted_order: List, true_order: List) -> float:
    """
    Compute Kendall's Tau correlation between predicted and true orderings.
    
    Returns value in [-1, 1] where:
    - 1 = perfect agreement
    - 0 = random ordering
    - -1 = perfect disagreement (reversed)
    """
    from scipy.stats import kendalltau
    
    if len(predicted_order) != len(true_order):
        raise ValueError("Orders must have same length")
    
    if len(predicted_order) < 2:
        return 1.0  # Perfect for single item
    
    # Convert to ranks
    pred_ranks = {item: i for i, item in enumerate(predicted_order)}
    true_ranks = {item: i for i, item in enumerate(true_order)}
    
    # Create rank arrays
    items = list(pred_ranks.keys())
    pred_array = [pred_ranks[item] for item in items]
    true_array = [true_ranks[item] for item in items]
    
    tau, _ = kendalltau(pred_array, true_array)
    
    return tau


def evaluate_timeline_ordering(
    predicted_timelines: Dict[str, List],
    true_timelines: Dict[str, List]
) -> Dict[str, float]:
    """
    Evaluate timeline ordering quality.
    
    Args:
        predicted_timelines: {cluster_id: [ordered article ids]}
        true_timelines: {topic_id: [ordered article ids]}
        
    Returns:
        Evaluation metrics
    """
    tau_scores = []
    
    for cluster_id, pred_order in predicted_timelines.items():
        if cluster_id in true_timelines:
            true_order = true_timelines[cluster_id]
            
            # Get common items
            common = set(pred_order) & set(true_order)
            
            if len(common) >= 2:
                pred_filtered = [x for x in pred_order if x in common]
                true_filtered = [x for x in true_order if x in common]
                
                tau = kendall_tau(pred_filtered, true_filtered)
                tau_scores.append(tau)
    
    if len(tau_scores) == 0:
        return {'kendall_tau_mean': 0, 'kendall_tau_std': 0, 'n_evaluated': 0}
    
    return {
        'kendall_tau_mean': np.mean(tau_scores),
        'kendall_tau_std': np.std(tau_scores),
        'n_evaluated': len(tau_scores)
    }


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_stage3(
    data_dir: str = './prepared_data',
    model_output: str = './models/stage3_threading.pkl'
) -> Tuple[EventThreadingClassifier, Dict[str, float]]:
    """
    Train Stage 3 event threading classifier.
    
    Args:
        data_dir: Directory with prepared data
        model_output: Where to save trained model
        
    Returns:
        (classifier, metrics)
    """
    print("="*60)
    print("STAGE 3: EVENT THREADING CLASSIFIER")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Load threading data
    print("\nLoading threading data...")
    train_df = pd.read_csv(data_path / 'threading_train.csv')
    val_df = pd.read_csv(data_path / 'threading_val.csv')
    test_df = pd.read_csv(data_path / 'threading_test.csv')
    
    print(f"Train: {len(train_df)} pairs")
    print(f"Val:   {len(val_df)} pairs")
    print(f"Test:  {len(test_df)} pairs")
    
    # Initialize and train classifier
    classifier = EventThreadingClassifier(model_type='xgboost')
    classifier.fit(train_df, val_df)
    
    # Evaluate
    metrics = classifier.evaluate(test_df)
    
    # Save model
    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_output)
    
    print("\n" + "="*60)
    print("STAGE 3 COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_output}")
    
    return classifier, metrics


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    classifier, metrics = train_stage3(
        data_dir='./prepared_data',
        model_output='./models/stage3_threading.pkl'
    )