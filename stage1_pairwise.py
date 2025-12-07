"""
Stage 1: Pairwise Relatedness Classifier (XGBoost)

This is an updated version of your original XGBoost code that:
1. Works with the prepared data from data_preparation.py
2. Saves the trained model for use in Stage 2
3. Includes a prediction function for scoring article pairs

Based on your original code, adapted for the pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import pickle
import tempfile
import shutil
from typing import Tuple, List, Optional
from pathlib import Path


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_tfidf_features(
    train_texts_a: pd.Series,
    train_texts_b: pd.Series,
    test_texts_a: pd.Series,
    test_texts_b: pd.Series,
    vectorizer: TfidfVectorizer = None,
    max_features: int = 5000
) -> Tuple[Tuple[csr_matrix, csr_matrix], Tuple[csr_matrix, csr_matrix], TfidfVectorizer]:
    """
    Create TF-IDF features for text pairs.
    
    Returns sparse matrices to save memory.
    
    If vectorizer is provided, uses it (for inference).
    Otherwise, fits a new vectorizer (for training).
    """
    print("Creating TF-IDF features...")
    
    # Preprocess
    train_a = train_texts_a.apply(preprocess_text)
    train_b = train_texts_b.apply(preprocess_text)
    test_a = test_texts_a.apply(preprocess_text)
    test_b = test_texts_b.apply(preprocess_text)
    
    # Fit or use existing vectorizer
    if vectorizer is None:
        train_all_texts = pd.concat([train_a, train_b])
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        vectorizer.fit(train_all_texts)
    
    # Transform - keep as sparse matrices
    train_tfidf_a = vectorizer.transform(train_a)
    train_tfidf_b = vectorizer.transform(train_b)
    test_tfidf_a = vectorizer.transform(test_a)
    test_tfidf_b = vectorizer.transform(test_b)
    
    print(f"TF-IDF features created: {train_tfidf_a.shape[1]} features per text (sparse format)")
    
    return (train_tfidf_a, train_tfidf_b), (test_tfidf_a, test_tfidf_b), vectorizer


def compute_similarity_features(
    texts_a: pd.Series,
    texts_b: pd.Series,
    tfidf_a: csr_matrix,
    tfidf_b: csr_matrix,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Compute hand-crafted similarity features between text pairs.
    
    Processes in batches to save memory.
    
    Features:
    - Jaccard similarity
    - Common word count
    - Word counts (a, b, diff, ratio)
    - Char lengths (a, b, diff, ratio)
    - Cosine similarity (from TF-IDF)
    - Euclidean distance (from TF-IDF)
    """
    print("Computing similarity features...")
    n_samples = len(texts_a)
    features = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_texts_a = texts_a.iloc[start_idx:end_idx]
        batch_texts_b = texts_b.iloc[start_idx:end_idx]
        
        # Convert only this batch to dense for TF-IDF operations
        batch_tfidf_a = tfidf_a[start_idx:end_idx].toarray()
        batch_tfidf_b = tfidf_b[start_idx:end_idx].toarray()
        
        batch_features = []
        for i in range(len(batch_texts_a)):
            text_a = preprocess_text(batch_texts_a.iloc[i])
            text_b = preprocess_text(batch_texts_b.iloc[i])
            
            # Word-level features
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            jaccard = intersection / union if union > 0 else 0
            common_words = intersection
            
            word_count_a = len(words_a)
            word_count_b = len(words_b)
            word_count_diff = abs(word_count_a - word_count_b)
            word_count_ratio = min(word_count_a, word_count_b) / max(word_count_a, word_count_b) if max(word_count_a, word_count_b) > 0 else 0
            
            # Character-level features
            char_len_a = len(text_a)
            char_len_b = len(text_b)
            char_len_diff = abs(char_len_a - char_len_b)
            char_len_ratio = min(char_len_a, char_len_b) / max(char_len_a, char_len_b) if max(char_len_a, char_len_b) > 0 else 0
            
            # TF-IDF based features
            vec_a = batch_tfidf_a[i]
            vec_b = batch_tfidf_b[i]
            
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            cosine_sim = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
            euclidean_dist = np.linalg.norm(vec_a - vec_b)
            
            batch_features.append([
                jaccard, common_words,
                word_count_a, word_count_b, word_count_diff, word_count_ratio,
                char_len_a, char_len_b, char_len_diff, char_len_ratio,
                cosine_sim, euclidean_dist
            ])
        
        features.extend(batch_features)
        
        if end_idx % 100000 == 0 or end_idx == n_samples:
            print(f"  Processed {end_idx}/{n_samples} pairs...")
    
    return np.array(features)


def create_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vectorizer: TfidfVectorizer = None,
    max_tfidf_features: int = 5000,
    batch_size: int = 10000,
    save_to_disk: bool = True,
    temp_dir: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], TfidfVectorizer]:
    """
    Create full feature matrix for train and test sets.
    
    Processes in batches to save memory. If save_to_disk is True, saves batches
    to disk and returns file paths instead of arrays.
    
    Returns:
        If save_to_disk: (train_file, test_file, vectorizer) - paths to .npy files
        Otherwise: (X_train, X_test, vectorizer) - numpy arrays
    """
    (train_tfidf_a, train_tfidf_b), (test_tfidf_a, test_tfidf_b), vectorizer = create_tfidf_features(
        train_df['text_a'], train_df['text_b'],
        test_df['text_a'], test_df['text_b'],
        vectorizer=vectorizer,
        max_features=max_tfidf_features
    )
    
    train_sim_features = compute_similarity_features(
        train_df['text_a'], train_df['text_b'],
        train_tfidf_a, train_tfidf_b,
        batch_size=batch_size
    )
    test_sim_features = compute_similarity_features(
        test_df['text_a'], test_df['text_b'],
        test_tfidf_a, test_tfidf_b,
        batch_size=batch_size
    )
    
    # Create temp directory if needed
    if save_to_disk:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix='xgboost_features_')
        else:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving feature batches to: {temp_dir}")
    
    # Concatenate: [tfidf_a | tfidf_b | similarity_features]
    # Convert sparse TF-IDF to dense in batches and save to disk
    print("Assembling final feature matrices...")
    
    # For train: process in batches and save to disk
    n_train = len(train_df)
    n_features = None
    train_batch_files = []
    
    for batch_idx, start_idx in enumerate(range(0, n_train, batch_size)):
        end_idx = min(start_idx + batch_size, n_train)
        batch_tfidf_a = train_tfidf_a[start_idx:end_idx].toarray()
        batch_tfidf_b = train_tfidf_b[start_idx:end_idx].toarray()
        batch_sim = train_sim_features[start_idx:end_idx]
        batch_X = np.hstack([batch_tfidf_a, batch_tfidf_b, batch_sim])
        
        if n_features is None:
            n_features = batch_X.shape[1]
        
        if save_to_disk:
            batch_file = Path(temp_dir) / f'train_batch_{batch_idx}.npy'
            np.save(batch_file, batch_X)
            train_batch_files.append(batch_file)
            del batch_X, batch_tfidf_a, batch_tfidf_b, batch_sim  # Free memory
        else:
            train_batch_files.append(batch_X)
        
        if end_idx % 100000 == 0 or end_idx == n_train:
            print(f"  Assembled {end_idx}/{n_train} training samples...")
    
    # For test: process in batches and save to disk
    n_test = len(test_df)
    test_batch_files = []
    
    for batch_idx, start_idx in enumerate(range(0, n_test, batch_size)):
        end_idx = min(start_idx + batch_size, n_test)
        batch_tfidf_a = test_tfidf_a[start_idx:end_idx].toarray()
        batch_tfidf_b = test_tfidf_b[start_idx:end_idx].toarray()
        batch_sim = test_sim_features[start_idx:end_idx]
        batch_X = np.hstack([batch_tfidf_a, batch_tfidf_b, batch_sim])
        
        if save_to_disk:
            batch_file = Path(temp_dir) / f'test_batch_{batch_idx}.npy'
            np.save(batch_file, batch_X)
            test_batch_files.append(batch_file)
            del batch_X, batch_tfidf_a, batch_tfidf_b, batch_sim  # Free memory
        else:
            test_batch_files.append(batch_X)
        
        if end_idx % 100000 == 0 or end_idx == n_test:
            print(f"  Assembled {end_idx}/{n_test} test samples...")
    
    if save_to_disk:
        # Save metadata
        metadata = {
            'n_features': n_features,
            'n_train': n_train,
            'n_test': n_test,
            'batch_size': batch_size,
            'train_batches': len(train_batch_files),
            'test_batches': len(test_batch_files)
        }
        metadata_file = Path(temp_dir) / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Feature matrices saved to disk. Train: {n_train} samples, Test: {n_test} samples, Features: {n_features}")
        return temp_dir, None, vectorizer
    else:
        # Old behavior: stack all batches in memory
        X_train = np.vstack(train_batch_files)
        X_test = np.vstack(test_batch_files)
        print(f"Final feature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, vectorizer


# =============================================================================
# MODEL TRAINING
# =============================================================================

def load_features_from_disk(temp_dir: str, split: str = 'train', batch_size: int = 10000):
    """
    Generator that loads feature batches from disk one at a time.
    
    Args:
        temp_dir: Directory containing feature batches
        split: 'train' or 'test'
        batch_size: Batch size used when creating features
    """
    temp_path = Path(temp_dir)
    # Try .npy files first, then .npz (compressed)
    batch_files_npy = sorted(temp_path.glob(f'{split}_batch_*.npy'))
    batch_files_npz = sorted(temp_path.glob(f'{split}_batch_*.npz'))
    batch_files = batch_files_npy + batch_files_npz
    
    for batch_file in batch_files:
        if batch_file.suffix == '.npz':
            # Load compressed format
            data = np.load(batch_file)
            batch_X = data['data']
            data.close()
        else:
            batch_X = np.load(batch_file)  # Load into memory (needed for stacking)
        yield batch_X


def train_model(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    use_early_stopping: bool = True,
    train_data_dir: Optional[str] = None,
    val_data_dir: Optional[str] = None
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with optional early stopping.
    
    Can train from in-memory arrays or from disk-based batches.
    If train_data_dir is provided, loads features from disk incrementally.
    """
    print("Training XGBoost model...")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if use_early_stopping and (X_val is not None or val_data_dir is not None):
        params['early_stopping_rounds'] = 10
    
    model = xgb.XGBClassifier(**params)
    
    # Train from disk if needed
    if train_data_dir is not None:
        print("Loading training data from disk in batches...")
        # Load metadata
        metadata_file = Path(train_data_dir) / 'metadata.pkl'
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load batches and train incrementally
        train_batches = list(load_features_from_disk(train_data_dir, 'train'))
        X_train_full = np.vstack(train_batches)
        
        # Load validation if from disk
        if val_data_dir is not None:
            val_batches = list(load_features_from_disk(val_data_dir, 'test'))
            X_val = np.vstack(val_batches)
        
        fit_params = {}
        if use_early_stopping and X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        model.fit(X_train_full, y_train, **fit_params)
        
        # Clean up memory
        del X_train_full, train_batches
        if val_data_dir is not None:
            del val_batches
    else:
        # Original in-memory training
        fit_params = {}
        if use_early_stopping and X_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        model.fit(X_train, y_train, **fit_params)
    
    print("Model training completed!")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate model and print metrics."""
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
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
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

def save_model(model: xgb.XGBClassifier, vectorizer: TfidfVectorizer, filepath: str):
    """Save model and vectorizer for later use."""
    print(f"Saving model to {filepath}...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved successfully!")


def load_model(filepath: str) -> Tuple[xgb.XGBClassifier, TfidfVectorizer]:
    """Load saved model and vectorizer."""
    print(f"Loading model from {filepath}...")
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['vectorizer']


# =============================================================================
# PREDICTION INTERFACE (for Stage 2)
# =============================================================================

class PairwiseClassifier:
    """
    Wrapper class for easy prediction on new article pairs.
    
    This is what Stage 2 (Leiden clustering) will use to score pairs.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.vectorizer = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model."""
        self.model, self.vectorizer = load_model(model_path)
    
    def predict_pair(self, text_a: str, text_b: str) -> Tuple[int, float]:
        """
        Predict if two texts are related.
        
        Returns:
            (prediction, probability)
            prediction: 0 (unrelated) or 1 (related)
            probability: P(related)
        """
        # Create single-row dataframe
        df = pd.DataFrame({'text_a': [text_a], 'text_b': [text_b]})
        
        # Get features
        texts_a = df['text_a'].apply(preprocess_text)
        texts_b = df['text_b'].apply(preprocess_text)
        
        tfidf_a = self.vectorizer.transform(texts_a).toarray()
        tfidf_b = self.vectorizer.transform(texts_b).toarray()
        
        # For single pair, convert to dense for similarity computation
        sim_features = compute_similarity_features(
            df['text_a'], df['text_b'], 
            csr_matrix(tfidf_a), csr_matrix(tfidf_b),
            batch_size=1
        )
        
        X = np.hstack([tfidf_a, tfidf_b, sim_features])
        
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0, 1]
        
        return int(pred), float(prob)
    
    def predict_pairs_batch(
        self,
        texts_a: List[str],
        texts_b: List[str],
        batch_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple pairs.
        
        Returns:
            (predictions, probabilities)
        """
        df = pd.DataFrame({'text_a': texts_a, 'text_b': texts_b})
        
        all_preds = []
        all_probs = []
        
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]
            
            texts_a_batch = batch_df['text_a'].apply(preprocess_text)
            texts_b_batch = batch_df['text_b'].apply(preprocess_text)
            
            tfidf_a_sparse = self.vectorizer.transform(texts_a_batch)
            tfidf_b_sparse = self.vectorizer.transform(texts_b_batch)
            
            sim_features = compute_similarity_features(
                batch_df['text_a'], batch_df['text_b'],
                tfidf_a_sparse, tfidf_b_sparse,
                batch_size=batch_size
            )
            
            # Convert to dense only for this batch
            tfidf_a = tfidf_a_sparse.toarray()
            tfidf_b = tfidf_b_sparse.toarray()
            
            X = np.hstack([tfidf_a, tfidf_b, sim_features])
            
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)[:, 1]
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            if end % 10000 == 0:
                print(f"  Processed {end}/{len(df)} pairs...")
        
        return np.array(all_preds), np.array(all_probs)


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_stage1(
    data_dir: str = './prepared_data',
    model_output: str = './models/stage1_xgboost.pkl',
    max_tfidf_features: int = 5000
) -> Tuple[xgb.XGBClassifier, TfidfVectorizer, dict]:
    """
    Main function to train Stage 1 pairwise classifier.
    
    Args:
        data_dir: Directory containing prepared data
        model_output: Where to save trained model
        max_tfidf_features: Number of TF-IDF features
        
    Returns:
        (model, vectorizer, metrics)
    """
    print("="*60)
    print("STAGE 1: PAIRWISE RELATEDNESS CLASSIFIER")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Load prepared data
    print("\nLoading prepared data...")
    train_df = pd.read_csv(data_path / 'xgboost_train.csv')
    val_df = pd.read_csv(data_path / 'xgboost_val.csv')
    test_df = pd.read_csv(data_path / 'xgboost_test.csv')
    
    print(f"Train: {len(train_df)} pairs")
    print(f"Val:   {len(val_df)} pairs")
    print(f"Test:  {len(test_df)} pairs")
    
    # Drop any NaN values
    train_df = train_df.dropna(subset=['text_a', 'text_b', 'label'])
    val_df = val_df.dropna(subset=['text_a', 'text_b', 'label'])
    test_df = test_df.dropna(subset=['text_a', 'text_b', 'label'])
    
    # Fit vectorizer on combined train+val data
    print("Fitting TF-IDF vectorizer...")
    combined_texts = pd.concat([
        train_df['text_a'], train_df['text_b'],
        val_df['text_a'], val_df['text_b']
    ], ignore_index=True).apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    vectorizer.fit(combined_texts)
    del combined_texts  # Free memory
    
    # Use disk-based storage to save memory
    # Use project directory instead of system temp to avoid space issues
    data_path_obj = Path(data_dir)
    temp_dir = data_path_obj / 'temp_features'
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Process train, val, and test separately
        # Create a helper function to process a single dataset
        def process_single_dataset(df, dataset_name, temp_subdir):
            """Process a single dataset and save to disk."""
            # Create directory if it doesn't exist
            Path(temp_subdir).mkdir(parents=True, exist_ok=True)
            
            (tfidf_a, tfidf_b), _, _ = create_tfidf_features(
                df['text_a'], df['text_b'],
                df['text_a'], df['text_b'],  # Dummy for test
                vectorizer=vectorizer,
                max_features=max_tfidf_features
            )
            
            sim_features = compute_similarity_features(
                df['text_a'], df['text_b'],
                tfidf_a, tfidf_b,
                batch_size=10000
            )
            
            # Save batches to disk
            # Use smaller batch size for saving to avoid very large files (~800MB per batch)
            save_batch_size = 5000  # Smaller batches for disk I/O
            n_samples = len(df)
            n_features = None
            
            for batch_idx, start_idx in enumerate(range(0, n_samples, save_batch_size)):
                end_idx = min(start_idx + save_batch_size, n_samples)
                batch_tfidf_a = tfidf_a[start_idx:end_idx].toarray()
                batch_tfidf_b = tfidf_b[start_idx:end_idx].toarray()
                batch_sim = sim_features[start_idx:end_idx]
                batch_X = np.hstack([batch_tfidf_a, batch_tfidf_b, batch_sim])
                
                if n_features is None:
                    n_features = batch_X.shape[1]
                
                batch_file = Path(temp_subdir) / f'train_batch_{batch_idx}.npy'
                
                # Try to save with error handling
                try:
                    # Ensure contiguous array for efficient I/O
                    if not batch_X.flags['C_CONTIGUOUS']:
                        batch_X = np.ascontiguousarray(batch_X)
                    np.save(batch_file, batch_X, allow_pickle=False)
                except (OSError, IOError) as e:
                    # If save fails, try with compression
                    print(f"  Warning: Failed to save batch {batch_idx} ({batch_X.nbytes / 1e6:.1f}MB), trying compressed format...")
                    try:
                        # Try compressed format (smaller files)
                        batch_file_compressed = Path(temp_subdir) / f'train_batch_{batch_idx}.npz'
                        np.savez_compressed(batch_file_compressed, data=batch_X)
                        print(f"  Saved batch {batch_idx} in compressed format")
                    except Exception as e2:
                        error_msg = (
                            f"Failed to save batch {batch_idx} to {batch_file}.\n"
                            f"Original error: {e}\n"
                            f"Compressed save also failed: {e2}\n"
                            f"Batch size: {batch_X.shape}, Memory: {batch_X.nbytes / 1e9:.2f}GB\n"
                            f"Check disk space and permissions."
                        )
                        raise IOError(error_msg)
                
                del batch_X, batch_tfidf_a, batch_tfidf_b, batch_sim
                
                if end_idx % 100000 == 0 or end_idx == n_samples:
                    print(f"  Processed {end_idx}/{n_samples} {dataset_name} samples...")
            
            return temp_subdir
        
        print("\nProcessing training data...")
        train_data_dir = process_single_dataset(train_df, 'training', Path(temp_dir) / 'train')
        
        print("\nProcessing validation data...")
        val_data_dir = process_single_dataset(val_df, 'validation', Path(temp_dir) / 'val')
        
        # Only process test data if we have space - we can evaluate later
        # For now, skip test data processing to save disk space
        print("\nSkipping test data processing to save disk space...")
        print("  (Test data will be processed during evaluation if needed)")
        test_data_dir = None
        
        # Load batches and stack efficiently
        # For very large datasets, we load in chunks and stack incrementally
        print("\nLoading training batches...")
        train_batches = list(load_features_from_disk(train_data_dir, 'train'))
        print(f"  Stacking {len(train_batches)} training batches...")
        X_train = np.vstack(train_batches)
        del train_batches  # Free memory immediately
        
        print("Loading validation batches...")
        val_batches = list(load_features_from_disk(val_data_dir, 'train'))
        print(f"  Stacking {len(val_batches)} validation batches...")
        X_val = np.vstack(val_batches)
        del val_batches  # Free memory immediately
        
        # Process test data on-the-fly for evaluation (don't save to disk)
        print("\nProcessing test data on-the-fly for evaluation...")
        (test_tfidf_a, test_tfidf_b), _, _ = create_tfidf_features(
            test_df['text_a'], test_df['text_b'],
            test_df['text_a'], test_df['text_b'],
            vectorizer=vectorizer,
            max_features=max_tfidf_features
        )
        test_sim_features = compute_similarity_features(
            test_df['text_a'], test_df['text_b'],
            test_tfidf_a, test_tfidf_b,
            batch_size=10000
        )
        
        # Process test in smaller batches to avoid memory issues
        print("Assembling test features in batches...")
        n_test = len(test_df)
        test_batch_size = 5000
        test_batches = []
        for start_idx in range(0, n_test, test_batch_size):
            end_idx = min(start_idx + test_batch_size, n_test)
            batch_tfidf_a = test_tfidf_a[start_idx:end_idx].toarray()
            batch_tfidf_b = test_tfidf_b[start_idx:end_idx].toarray()
            batch_sim = test_sim_features[start_idx:end_idx]
            test_batches.append(np.hstack([batch_tfidf_a, batch_tfidf_b, batch_sim]))
            if end_idx % 100000 == 0 or end_idx == n_test:
                print(f"  Assembled {end_idx}/{n_test} test samples...")
        X_test = np.vstack(test_batches)
        del test_batches, test_tfidf_a, test_tfidf_b, test_sim_features
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, use_early_stopping=True)
        
        # Clean up temp files after training
        print(f"\nCleaning up temporary files in {temp_dir}...")
        try:
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up successfully.")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temp directory {temp_dir}: {cleanup_error}")
            print("You may need to manually delete it later.")
        
    except Exception as e:
        # Clean up on error (but keep files for debugging if space allows)
        print(f"\nError occurred. Temp files are in: {temp_dir}")
        print("You may want to clean them up manually to free disk space.")
        raise e
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, vectorizer, model_output)
    
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_output}")
    
    return model, vectorizer, metrics


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    model, vectorizer, metrics = train_stage1(
        data_dir='./prepared_data',
        model_output='./models/stage1_xgboost.pkl'
    )