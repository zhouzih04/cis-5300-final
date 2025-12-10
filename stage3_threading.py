"""
Stage 3: Event Threading for Temporal Ordering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import pickle
import re
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


TEMPORAL_RELATIONS = {
    'SAME_DAY': 0,           # Same day coverage of same event
    'IMMEDIATE_UPDATE': 1,    # 1-2 days: Direct follow-up
    'SHORT_TERM_DEV': 2,      # 3-7 days: Story development
    'LONG_TERM_DEV': 3,       # 8-30 days: Ongoing story
    'DISTANT_RELATED': 4      # 30+ days: Same topic, distant events
}


def extract_numbers(text: str) -> set:
    """Extract all numbers from text."""
    return set(re.findall(r'\b\d+\b', str(text)))


def extract_capitalized_words(text: str) -> set:
    """Extract capitalized words."""
    if pd.isna(text):
        return set()
    words = re.findall(r'\b[A-Z][a-z]+\b', str(text))
    return set(w.lower() for w in words)


def get_bigrams(text: str) -> set:
    """Extract word bigrams."""
    words = text.lower().split()
    if len(words) < 2:
        return set()
    return set(zip(words[:-1], words[1:]))


def extract_threading_features_fixed(
    title_a: str,
    title_b: str,
    tfidf_vectorizer: TfidfVectorizer = None
) -> Dict[str, float]:
    """Extract threading features."""
    features = {}
    
    text_a = str(title_a).lower().strip() if not pd.isna(title_a) else ""
    text_b = str(title_b).lower().strip() if not pd.isna(title_b) else ""
    orig_a = str(title_a) if not pd.isna(title_a) else ""
    orig_b = str(title_b) if not pd.isna(title_b) else ""
    
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    
    features['jaccard'] = intersection / union if union > 0 else 0
    features['common_words'] = intersection
    features['words_only_in_a'] = len(words_a - words_b)
    features['words_only_in_b'] = len(words_b - words_a)
    features['novelty_ratio'] = len(words_b - words_a) / len(words_b) if len(words_b) > 0 else 0
    
    features['len_a'] = len(words_a)
    features['len_b'] = len(words_b)
    features['len_diff'] = abs(len(words_a) - len(words_b))
    features['len_ratio'] = min(len(words_a), len(words_b)) / max(len(words_a), len(words_b)) if max(len(words_a), len(words_b)) > 0 else 0
    
    entities_a = extract_capitalized_words(orig_a)
    entities_b = extract_capitalized_words(orig_b)
    entity_intersection = len(entities_a & entities_b)
    entity_union = len(entities_a | entities_b)
    features['entity_overlap'] = entity_intersection / entity_union if entity_union > 0 else 0
    
    numbers_a = extract_numbers(orig_a)
    numbers_b = extract_numbers(orig_b)
    number_intersection = len(numbers_a & numbers_b)
    number_union = len(numbers_a | numbers_b)
    features['number_overlap'] = number_intersection / number_union if number_union > 0 else 0
    
    bigrams_a = get_bigrams(text_a)
    bigrams_b = get_bigrams(text_b)
    bigram_intersection = len(bigrams_a & bigrams_b)
    bigram_union = len(bigrams_a | bigrams_b)
    features['bigram_overlap'] = bigram_intersection / bigram_union if bigram_union > 0 else 0
    
    update_indicators = [
        'update', 'latest', 'new', 'now', 'breaking',
        'continues', 'ongoing', 'still', 'again', 'another',
        'more', 'further', 'additional'
    ]
    features['update_indicator_count'] = sum(1 for ind in update_indicators if ind in text_b)
    
    backward_indicators = [
        'after', 'following', 'since', 'aftermath',
        'response', 'reaction', 'result', 'consequence'
    ]
    features['backward_ref_count'] = sum(1 for ind in backward_indicators if ind in text_b)
    
    breaking_indicators = ['breaking', 'just in', 'alert', 'urgent', 'developing']
    features['is_breaking_a'] = 1 if any(ind in text_a for ind in breaking_indicators) else 0
    features['is_breaking_b'] = 1 if any(ind in text_b for ind in breaking_indicators) else 0
    
    if tfidf_vectorizer is not None:
        vec_a = tfidf_vectorizer.transform([text_a]).toarray()[0]
        vec_b = tfidf_vectorizer.transform([text_b]).toarray()[0]
        
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        features['tfidf_cosine'] = dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
        features['tfidf_euclidean'] = np.linalg.norm(vec_a - vec_b)
    else:
        features['tfidf_cosine'] = 0
        features['tfidf_euclidean'] = 0
    
    return features


# Feature names for reference
THREADING_FEATURE_NAMES = [
    # Lexical overlap
    'jaccard', 'common_words', 'words_only_in_a', 'words_only_in_b', 'novelty_ratio',
    # Length
    'len_a', 'len_b', 'len_diff', 'len_ratio',
    # Entity & number
    'entity_overlap', 'number_overlap', 'bigram_overlap',
    # Linguistic markers
    'update_indicator_count', 'backward_ref_count', 'is_breaking_a', 'is_breaking_b',
    # TF-IDF
    'tfidf_cosine', 'tfidf_euclidean'
]


def extract_features_batch(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer = None
) -> pd.DataFrame:
    """Extract features for batch."""
    
    features_list = []
    
    for idx, row in df.iterrows():
        features = extract_threading_features_fixed(
            row['text_a'], row['text_b'],
            tfidf_vectorizer
        )
        features_list.append(features)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} pairs...")
    
    features_df = pd.DataFrame(features_list)
    
    return features_df


class EventThreadingClassifier:
    """Event threading classifier."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """Initialize classifier."""
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
        """Train classifier."""
        all_texts = pd.concat([
            train_df['text_a'].apply(lambda x: str(x).lower() if not pd.isna(x) else ""),
            train_df['text_b'].apply(lambda x: str(x).lower() if not pd.isna(x) else "")
        ])
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(all_texts)
        
        X_train = extract_features_batch(train_df, self.tfidf_vectorizer)
        self.feature_columns = X_train.columns.tolist()
        y_train = self.label_encoder.fit_transform(train_df['temporal_relation'])
        
        print(f"Training: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Classes: {self.label_encoder.classes_}")
        
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
                n_jobs=-1,
                early_stopping_rounds=10 if val_df is not None else None
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
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            sorted_idx = np.argsort(importance)[::-1]
            print("Top 5 features:")
            for rank, idx in enumerate(sorted_idx[:5], 1):
                name = self.feature_columns[idx]
                print(f"  {name}: {importance[idx]:.4f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict temporal relations."""
        X = extract_features_batch(df, self.tfidf_vectorizer)
        X = X[self.feature_columns]
        
        y_pred = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(y_pred)
        
        return labels
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = extract_features_batch(df, self.tfidf_vectorizer)
        X = X[self.feature_columns]
        
        return self.model.predict_proba(X)
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate classifier."""
        y_true = test_df['temporal_relation'].values
        y_pred = self.predict(test_df)
        
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
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
        """Save model."""
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'version': 'fixed_no_leakage_v1'
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'EventThreadingClassifier':
        """Load model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.tfidf_vectorizer = model_data['tfidf_vectorizer']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        
        return classifier


class TimelineBuilder:
    """Timeline builder."""
    
    def __init__(self, threading_classifier: EventThreadingClassifier = None):
        """Initialize timeline builder."""
        self.classifier = threading_classifier
    
    def build_timeline(
        self,
        articles: pd.DataFrame,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """Build timeline."""
        if len(articles) == 0:
            return []
        
        if len(articles) == 1:
            return [self._article_to_dict(articles.iloc[0], position=0)]
        
        articles = articles.copy()
        articles['date'] = pd.to_datetime(articles['date'])
        
        if method == 'date':
            return self._build_by_date(articles)
        elif method == 'graph' and self.classifier is not None:
            return self._build_by_graph(articles)
        else:
            return self._build_hybrid(articles)
    
    def _article_to_dict(self, article: pd.Series, position: int, role: str = 'event') -> Dict:
        """Convert article to timeline entry."""
        date_val = article['date']
        if hasattr(date_val, 'date'):
            date_str = str(date_val.date())
        else:
            date_str = str(date_val)
        
        return {
            'position': position,
            'title': article['title'],
            'date': date_str,
            'role': role,
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
        """Build using threading predictions."""
        if not NETWORKX_AVAILABLE:
            return self._build_by_date(articles)
        
        # Create pairs
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
                        'idx_a': idx_a,
                        'idx_b': idx_b
                    })
        
        if len(pairs) == 0:
            return self._build_by_date(articles)
        
        pairs_df = pd.DataFrame(pairs)
        relations = self.classifier.predict(pairs_df)
        
        # Build graph
        G = nx.DiGraph()
        G.add_nodes_from(indices)
        
        for (_, row), relation in zip(pairs_df.iterrows(), relations):
            idx_a, idx_b = row['idx_a'], row['idx_b']
            date_a = articles.loc[idx_a, 'date']
            date_b = articles.loc[idx_b, 'date']
            
            if date_a <= date_b:
                earlier, later = idx_a, idx_b
            else:
                earlier, later = idx_b, idx_a
            
            if relation in ['IMMEDIATE_UPDATE', 'SHORT_TERM_DEV']:
                G.add_edge(earlier, later, relation=relation, weight=2)
            elif relation == 'SAME_DAY':
                G.add_edge(earlier, later, relation=relation, weight=1)
        
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            order = sorted(indices, key=lambda x: articles.loc[x, 'date'])
        
        timeline = []
        for i, idx in enumerate(order):
            article = articles.loc[idx]
            role = 'initial' if i == 0 else 'update'
            if G.in_degree(idx) > 1:
                role = 'development'
            timeline.append(self._article_to_dict(article, i, role))
        
        return timeline
    
    def _build_hybrid(self, articles: pd.DataFrame) -> List[Dict]:
        """Use dates as primary, content analysis for same-day ordering."""
        articles = articles.copy()
        articles['date_only'] = articles['date'].dt.date
        
        timeline = []
        position = 0
        
        for date, group in articles.groupby('date_only', sort=True):
            if len(group) == 1:
                article = group.iloc[0]
                role = 'initial' if position == 0 else 'update'
                timeline.append(self._article_to_dict(article, position, role))
                position += 1
            else:
                sub_timeline = self._order_same_day(group, position)
                timeline.extend(sub_timeline)
                position += len(sub_timeline)
        
        return timeline
    
    def _order_same_day(self, articles: pd.DataFrame, start_position: int) -> List[Dict]:
        """Order articles from the same day."""
        articles = articles.copy()
        
        def score_article(title):
            title_lower = str(title).lower()
            score = 0
            if any(ind in title_lower for ind in ['breaking', 'just in', 'alert']):
                score += 10
            score -= len(title_lower.split()) * 0.1
            return score
        
        articles['_score'] = articles['title'].apply(score_article)
        sorted_articles = articles.sort_values('_score', ascending=False)
        
        timeline = []
        for i, (_, article) in enumerate(sorted_articles.iterrows()):
            role = 'initial' if start_position + i == 0 else 'update'
            timeline.append(self._article_to_dict(article, start_position + i, role))
        
        return timeline


def kendall_tau(predicted_order: List, true_order: List) -> float:
    """Compute Kendall's Tau correlation."""
    from scipy.stats import kendalltau
    
    if len(predicted_order) != len(true_order):
        raise ValueError("Orders must have same length")
    
    if len(predicted_order) < 2:
        return 1.0
    
    pred_ranks = {item: i for i, item in enumerate(predicted_order)}
    true_ranks = {item: i for i, item in enumerate(true_order)}
    
    items = list(pred_ranks.keys())
    pred_array = [pred_ranks[item] for item in items]
    true_array = [true_ranks[item] for item in items]
    
    tau, _ = kendalltau(pred_array, true_array)
    
    return tau


def train_stage3(
    data_dir: str = './prepared_data',
    model_output: str = './models/stage3_threading.pkl'
) -> Tuple[EventThreadingClassifier, Dict[str, float]]:
    """Train Stage 3 event threading classifier."""
    data_path = Path(data_dir)
    
    print("Loading data...")
    train_df = pd.read_csv(data_path / 'threading_train.csv')
    val_df = pd.read_csv(data_path / 'threading_val.csv')
    test_df = pd.read_csv(data_path / 'threading_test.csv')
    
    print(f"Train: {len(train_df)} pairs, Val: {len(val_df)} pairs, Test: {len(test_df)} pairs")
    
    classifier = EventThreadingClassifier(model_type='xgboost')
    print("Training classifier...")
    classifier.fit(train_df, val_df)
    print("Evaluating classifier...")
    metrics = classifier.evaluate(test_df)
    
    print("Saving model...")
    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_output)
    
    return classifier, metrics


if __name__ == "__main__":
    classifier, metrics = train_stage3(
        data_dir='./prepared_data',
        model_output='./models/stage3_threading.pkl'
    )