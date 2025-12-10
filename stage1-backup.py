"""
Stage 1: Pairwise Relatedness Classifier (V3 - With Sentence Embeddings)

This version combines:
1. 25 hand-crafted features (lexical, entity, structural)
2. 4 sentence embedding features (cosine, euclidean, manhattan, dot)
   = 29 total features

Expected accuracy: 82-90%
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import pickle
from typing import Tuple, List, Optional, Set
from pathlib import Path
import warnings


# =============================================================================
# CHECK FOR SENTENCE TRANSFORMERS
# =============================================================================

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✓ sentence-transformers is available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not installed. Run: pip install sentence-transformers")
    print("  Falling back to TF-IDF features only.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Stage1Config:
    """Configuration for Stage 1."""
    
    def __init__(
        self,
        max_train_samples: int = 150000,      # Increased from 100K
        max_val_samples: int = 15000,
        max_test_samples: int = 50000,
        tfidf_max_features: int = 5000,
        batch_size: int = 5000,
        use_sentence_embeddings: bool = True,  # ON by default now
        embedding_model: str = 'all-MiniLM-L6-v2',  # Fast and good
        random_seed: int = 42
    ):
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.tfidf_max_features = tfidf_max_features
        self.batch_size = batch_size
        self.use_sentence_embeddings = use_sentence_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.embedding_model = embedding_model
        self.random_seed = random_seed


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def extract_numbers(text: str) -> Set[str]:
    """Extract all numbers from text."""
    return set(re.findall(r'\b\d+\b', text))


def extract_capitalized_words(text: str) -> Set[str]:
    """Extract capitalized words (proper nouns)."""
    if pd.isna(text):
        return set()
    words = re.findall(r'\b[A-Z][a-z]+\b', str(text))
    return set(w.lower() for w in words)


def get_bigrams(text: str) -> Set[Tuple[str, str]]:
    """Extract word bigrams."""
    words = text.split()
    if len(words) < 2:
        return set()
    return set(zip(words[:-1], words[1:]))


def get_first_n_words(text: str, n: int = 3) -> List[str]:
    """Get first N words."""
    words = text.split()
    return words[:n] if len(words) >= n else words


# =============================================================================
# SENTENCE EMBEDDINGS
# =============================================================================

class EmbeddingCache:
    """Cache for sentence embeddings to avoid recomputation."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.model_name = model_name
        self.cache = {}
    
    def load_model(self):
        """Lazy load the model."""
        if self.model is None:
            print(f"Loading sentence embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts, using cache when possible."""
        model = self.load_model()
        
        # Find which texts need encoding
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            if text not in self.cache:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode new texts
        if texts_to_encode:
            new_embeddings = model.encode(
                texts_to_encode, 
                batch_size=batch_size,
                show_progress_bar=len(texts_to_encode) > 1000
            )
            for text, emb in zip(texts_to_encode, new_embeddings):
                self.cache[text] = emb
        
        # Gather all embeddings
        result = np.array([self.cache[text] for text in texts])
        return result


def compute_embedding_similarity(
    texts_a: List[str],
    texts_b: List[str],
    embedding_cache: EmbeddingCache
) -> np.ndarray:
    """
    Compute similarity features from sentence embeddings.
    
    Returns 4 features per pair:
    - cosine_similarity
    - euclidean_distance (normalized)
    - manhattan_distance (normalized)
    - dot_product (normalized)
    """
    print(f"Computing embedding similarities for {len(texts_a)} pairs...")
    
    # Encode all texts
    all_texts = list(texts_a) + list(texts_b)
    all_embeddings = embedding_cache.encode(all_texts)
    
    emb_a = all_embeddings[:len(texts_a)]
    emb_b = all_embeddings[len(texts_a):]
    
    features = []
    for i in range(len(texts_a)):
        vec_a = emb_a[i]
        vec_b = emb_b[i]
        
        # Cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
        
        # Euclidean distance (normalized to 0-1 range approximately)
        euclidean = np.linalg.norm(vec_a - vec_b) / 10.0  # Normalize
        
        # Manhattan distance (normalized)
        manhattan = np.sum(np.abs(vec_a - vec_b)) / 100.0  # Normalize
        
        # Dot product (normalized)
        dot = np.dot(vec_a, vec_b) / 10.0  # Normalize
        
        features.append([cosine, euclidean, manhattan, dot])
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# HAND-CRAFTED FEATURES (25 features)
# =============================================================================

def compute_handcrafted_features(
    texts_a: pd.Series,
    texts_b: pd.Series,
    vectorizer: TfidfVectorizer,
    batch_size: int = 5000
) -> np.ndarray:
    """Compute 25 hand-crafted features."""
    n_samples = len(texts_a)
    all_features = []
    
    original_texts_a = texts_a.copy()
    original_texts_b = texts_b.copy()
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        batch_orig_a = original_texts_a.iloc[start_idx:end_idx]
        batch_orig_b = original_texts_b.iloc[start_idx:end_idx]
        batch_texts_a = texts_a.iloc[start_idx:end_idx].apply(preprocess_text)
        batch_texts_b = texts_b.iloc[start_idx:end_idx].apply(preprocess_text)
        
        batch_tfidf_a = vectorizer.transform(batch_texts_a).toarray()
        batch_tfidf_b = vectorizer.transform(batch_texts_b).toarray()
        
        batch_features = []
        
        for i in range(len(batch_texts_a)):
            text_a = batch_texts_a.iloc[i]
            text_b = batch_texts_b.iloc[i]
            orig_a = str(batch_orig_a.iloc[i]) if not pd.isna(batch_orig_a.iloc[i]) else ""
            orig_b = str(batch_orig_b.iloc[i]) if not pd.isna(batch_orig_b.iloc[i]) else ""
            
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            
            # Basic lexical (1-6)
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            jaccard = intersection / union if union > 0 else 0
            common_words = intersection
            word_count_a = len(words_a)
            word_count_b = len(words_b)
            word_count_diff = abs(word_count_a - word_count_b)
            word_count_ratio = min(word_count_a, word_count_b) / max(word_count_a, word_count_b) if max(word_count_a, word_count_b) > 0 else 0
            
            # Character-level (7-10)
            char_len_a = len(text_a)
            char_len_b = len(text_b)
            char_len_diff = abs(char_len_a - char_len_b)
            char_len_ratio = min(char_len_a, char_len_b) / max(char_len_a, char_len_b) if max(char_len_a, char_len_b) > 0 else 0
            
            # TF-IDF similarity (11-12)
            vec_a = batch_tfidf_a[i]
            vec_b = batch_tfidf_b[i]
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            cosine_sim = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
            euclidean_dist = np.linalg.norm(vec_a - vec_b)
            
            # Advanced lexical (13-19)
            bigrams_a = get_bigrams(text_a)
            bigrams_b = get_bigrams(text_b)
            bigram_intersection = len(bigrams_a & bigrams_b)
            bigram_union = len(bigrams_a | bigrams_b)
            bigram_overlap = bigram_intersection / bigram_union if bigram_union > 0 else 0
            
            first_a = text_a.split()[0] if text_a.split() else ""
            first_b = text_b.split()[0] if text_b.split() else ""
            first_word_match = 1 if first_a == first_b and first_a != "" else 0
            
            first3_a = set(get_first_n_words(text_a, 3))
            first3_b = set(get_first_n_words(text_b, 3))
            first3_intersection = len(first3_a & first3_b)
            first_3_words_overlap = first3_intersection / 3 if first3_a or first3_b else 0
            
            numbers_a = extract_numbers(orig_a)
            numbers_b = extract_numbers(orig_b)
            number_intersection = len(numbers_a & numbers_b)
            number_union = len(numbers_a | numbers_b)
            number_overlap = number_intersection / number_union if number_union > 0 else 0
            
            entities_a = extract_capitalized_words(orig_a)
            entities_b = extract_capitalized_words(orig_b)
            entity_intersection = len(entities_a & entities_b)
            entity_union = len(entities_a | entities_b)
            entity_overlap = entity_intersection / entity_union if entity_union > 0 else 0
            
            unique_words_a = len(words_a - words_b)
            unique_words_b = len(words_b - words_a)
            
            # Structural (20-25)
            both_have_numbers = 1 if numbers_a and numbers_b else 0
            has_quote_a = 1 if '"' in orig_a or "'" in orig_a else 0
            has_quote_b = 1 if '"' in orig_b or "'" in orig_b else 0
            both_have_quotes = 1 if has_quote_a and has_quote_b else 0
            
            max_len = max(char_len_a, char_len_b)
            length_similarity = 1 - (char_len_diff / max_len) if max_len > 0 else 1
            
            prefix_match_len = 0
            min_len = min(len(text_a), len(text_b))
            for j in range(min_len):
                if text_a[j] == text_b[j]:
                    prefix_match_len += 1
                else:
                    break
            prefix_match_len = prefix_match_len / max_len if max_len > 0 else 0
            
            containment_a_in_b = len(words_a & words_b) / len(words_a) if len(words_a) > 0 else 0
            containment_b_in_a = len(words_a & words_b) / len(words_b) if len(words_b) > 0 else 0
            
            batch_features.append([
                jaccard, common_words, word_count_a, word_count_b, word_count_diff, word_count_ratio,
                char_len_a, char_len_b, char_len_diff, char_len_ratio,
                cosine_sim, euclidean_dist,
                bigram_overlap, first_word_match, first_3_words_overlap,
                number_overlap, entity_overlap, unique_words_a, unique_words_b,
                both_have_numbers, both_have_quotes, length_similarity,
                prefix_match_len, containment_a_in_b, containment_b_in_a
            ])
        
        all_features.extend(batch_features)
        del batch_tfidf_a, batch_tfidf_b, batch_features
        
        if (end_idx) % 20000 == 0 or end_idx == n_samples:
            print(f"  Processed {end_idx}/{n_samples} pairs...")
    
    return np.array(all_features, dtype=np.float32)


FEATURE_NAMES = [
    'jaccard', 'common_words', 'word_count_a', 'word_count_b', 'word_count_diff', 'word_count_ratio',
    'char_len_a', 'char_len_b', 'char_len_diff', 'char_len_ratio',
    'tfidf_cosine', 'tfidf_euclidean',
    'bigram_overlap', 'first_word_match', 'first_3_words_overlap',
    'number_overlap', 'entity_overlap', 'unique_words_a', 'unique_words_b',
    'both_have_numbers', 'both_have_quotes', 'length_similarity',
    'prefix_match_len', 'containment_a_in_b', 'containment_b_in_a',
    # Embedding features (if used)
    'emb_cosine', 'emb_euclidean', 'emb_manhattan', 'emb_dot'
]


# =============================================================================
# DATA SAMPLING
# =============================================================================

def sample_balanced_data(df: pd.DataFrame, max_samples: int, random_seed: int = 42) -> pd.DataFrame:
    """Sample data while maintaining class balance."""
    if len(df) <= max_samples:
        return df
    
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    samples_per_class = max_samples // 2
    
    pos_sampled = pos_df.sample(n=min(samples_per_class, len(pos_df)), random_state=random_seed)
    neg_sampled = neg_df.sample(n=min(samples_per_class, len(neg_df)), random_state=random_seed)
    
    result = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"  Sampled {len(result)} pairs (was {len(df)})")
    print(f"  Class balance: {(result['label']==1).mean():.1%} positive")
    
    return result


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None) -> xgb.XGBClassifier:
    """Train XGBoost with optimal parameters."""
    print("\nTraining XGBoost model...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=8,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15 if X_val is not None else None
    )
    
    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False
    
    model.fit(X_train, y_train, **fit_params)
    print("  Training complete!")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate and print metrics."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos   {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unrelated', 'Related']))
    
    return metrics


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_model(model, vectorizer, filepath, config=None, embedding_cache=None):
    """Save model and all components."""
    print(f"\nSaving model to {filepath}...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'version': 'v3_with_embeddings',
        'feature_names': FEATURE_NAMES,
        'config': config,
        'uses_embeddings': config.use_sentence_embeddings if config else False,
        'embedding_model': config.embedding_model if config else None
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved!")


def load_model(filepath: str):
    """Load model and vectorizer."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['vectorizer'], model_data.get('uses_embeddings', False), model_data.get('embedding_model', 'all-MiniLM-L6-v2')


# =============================================================================
# PREDICTION INTERFACE
# =============================================================================

class PairwiseClassifier:
    """Wrapper for prediction on article pairs."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.vectorizer = None
        self.uses_embeddings = False
        self.embedding_cache = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model."""
        self.model, self.vectorizer, self.uses_embeddings, embedding_model = load_model(model_path)
        if self.uses_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_cache = EmbeddingCache(embedding_model)
        print(f"Loaded model from {model_path}")
        print(f"  Uses embeddings: {self.uses_embeddings}")
    
    def predict_pair(self, text_a: str, text_b: str) -> Tuple[int, float]:
        """Predict if two texts are related."""
        df = pd.DataFrame({'text_a': [text_a], 'text_b': [text_b]})
        
        # Hand-crafted features
        features = compute_handcrafted_features(df['text_a'], df['text_b'], self.vectorizer, batch_size=1)
        
        # Embedding features
        if self.uses_embeddings and self.embedding_cache:
            emb_features = compute_embedding_similarity([text_a], [text_b], self.embedding_cache)
            features = np.hstack([features, emb_features])
        
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0, 1]
        
        return int(pred), float(prob)
    
    def predict_pairs_batch(self, texts_a: List[str], texts_b: List[str], batch_size: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        df = pd.DataFrame({'text_a': texts_a, 'text_b': texts_b})
        
        all_preds = []
        all_probs = []
        
        n_samples = len(df)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_df = df.iloc[start:end]
            
            features = compute_handcrafted_features(
                batch_df['text_a'].reset_index(drop=True),
                batch_df['text_b'].reset_index(drop=True),
                self.vectorizer, batch_size=batch_size
            )
            
            if self.uses_embeddings and self.embedding_cache:
                emb_features = compute_embedding_similarity(
                    batch_df['text_a'].tolist(),
                    batch_df['text_b'].tolist(),
                    self.embedding_cache
                )
                features = np.hstack([features, emb_features])
            
            preds = self.model.predict(features)
            probs = self.model.predict_proba(features)[:, 1]
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            if end % 20000 == 0 or end == n_samples:
                print(f"  Predicted {end}/{n_samples} pairs...")
        
        return np.array(all_preds), np.array(all_probs)


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_stage1(
    data_dir: str = './prepared_data',
    model_output: str = './models/stage1_xgboost.pkl',
    config: Stage1Config = None
) -> Tuple[xgb.XGBClassifier, TfidfVectorizer, dict]:
    """Train Stage 1 pairwise classifier with embeddings."""
    config = config or Stage1Config()
    
    print("="*60)
    print("STAGE 1: PAIRWISE CLASSIFIER (V3 + Embeddings)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Max train samples: {config.max_train_samples:,}")
    print(f"  Use sentence embeddings: {config.use_sentence_embeddings}")
    if config.use_sentence_embeddings:
        print(f"  Embedding model: {config.embedding_model}")
    print(f"  Features: 25 hand-crafted + 4 embedding = 29 total")
    
    data_path = Path(data_dir)
    
    # Load data
    print("\nLoading prepared data...")
    train_df = pd.read_csv(data_path / 'xgboost_train.csv')
    val_df = pd.read_csv(data_path / 'xgboost_val.csv')
    test_df = pd.read_csv(data_path / 'xgboost_test.csv')
    
    print(f"  Raw train: {len(train_df):,} pairs")
    print(f"  Raw val: {len(val_df):,} pairs")
    print(f"  Raw test: {len(test_df):,} pairs")
    
    # Clean
    train_df = train_df.dropna(subset=['text_a', 'text_b', 'label'])
    val_df = val_df.dropna(subset=['text_a', 'text_b', 'label'])
    test_df = test_df.dropna(subset=['text_a', 'text_b', 'label'])
    
    # Sample
    print("\nSampling data...")
    train_df = sample_balanced_data(train_df, config.max_train_samples, config.random_seed)
    val_df = sample_balanced_data(val_df, config.max_val_samples, config.random_seed)
    test_df = sample_balanced_data(test_df, config.max_test_samples, config.random_seed)
    
    # Fit TF-IDF
    print("\nFitting TF-IDF vectorizer...")
    all_texts = pd.concat([
        train_df['text_a'].apply(preprocess_text),
        train_df['text_b'].apply(preprocess_text)
    ])
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    vectorizer.fit(all_texts)
    del all_texts
    
    # Initialize embedding cache if needed
    embedding_cache = None
    if config.use_sentence_embeddings:
        embedding_cache = EmbeddingCache(config.embedding_model)
    
    # Compute features
    print("\nComputing hand-crafted features...")
    print("  Training:")
    X_train_hc = compute_handcrafted_features(train_df['text_a'], train_df['text_b'], vectorizer, config.batch_size)
    print("  Validation:")
    X_val_hc = compute_handcrafted_features(val_df['text_a'], val_df['text_b'], vectorizer, config.batch_size)
    print("  Test:")
    X_test_hc = compute_handcrafted_features(test_df['text_a'], test_df['text_b'], vectorizer, config.batch_size)
    
    # Compute embedding features
    if config.use_sentence_embeddings:
        print("\nComputing sentence embedding features...")
        print("  Training:")
        X_train_emb = compute_embedding_similarity(train_df['text_a'].tolist(), train_df['text_b'].tolist(), embedding_cache)
        print("  Validation:")
        X_val_emb = compute_embedding_similarity(val_df['text_a'].tolist(), val_df['text_b'].tolist(), embedding_cache)
        print("  Test:")
        X_test_emb = compute_embedding_similarity(test_df['text_a'].tolist(), test_df['text_b'].tolist(), embedding_cache)
        
        X_train = np.hstack([X_train_hc, X_train_emb])
        X_val = np.hstack([X_val_hc, X_val_emb])
        X_test = np.hstack([X_test_hc, X_test_emb])
    else:
        X_train = X_train_hc
        X_val = X_val_hc
        X_test = X_test_hc
    
    y_train = train_df['label'].values.astype(np.int32)
    y_val = val_df['label'].values.astype(np.int32)
    y_test = test_df['label'].values.astype(np.int32)
    
    print(f"\nFeature matrices:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save
    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, vectorizer, model_output, config)
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance = model.feature_importances_
    feature_names = FEATURE_NAMES[:X_train.shape[1]]
    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank}. {name}: {importance[idx]:.4f}")
    
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE!")
    print("="*60)
    
    return model, vectorizer, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--model_output', type=str, default='./models/stage1_xgboost.pkl')
    parser.add_argument('--max_train', type=int, default=150000)
    parser.add_argument('--no_embeddings', action='store_true', help='Disable sentence embeddings')
    
    args = parser.parse_args()
    
    config = Stage1Config(
        max_train_samples=args.max_train,
        use_sentence_embeddings=not args.no_embeddings
    )
    
    model, vectorizer, metrics = train_stage1(
        data_dir=args.data_dir,
        model_output=args.model_output,
        config=config
    )