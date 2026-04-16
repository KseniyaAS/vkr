# /thesis_pipeline.py
"""Thesis-ready ticket categorization pipeline with CV and Transformer support."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional, Any
import pickle
import re
import time

import numpy as np
import pandas as pd
# import nltk  <-- MOVE INSIDE FUNCTIONS
# from nltk.corpus import stopwords <-- MOVE INSIDE FUNCTIONS
# from pymorphy3 import MorphAnalyzer <-- MOVE INSIDE FUNCTIONS
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.compose import ColumnTransformer


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Sentence Transformers for semantic embeddings
def get_sentence_transformer(model_name: str):
    """Lazy load SentenceTransformer only when needed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def preprocess_cleaned_cyr(text: object) -> str:
    """Keep only Cyrillic characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    return " ".join(text.split())

def preprocess_raw(text: object) -> str:
    """Basic normalization: lowercase, strip, and split."""
    if not isinstance(text, str):
        return ""
    return str(text).lower().strip()

DEFAULT_TEXT_COLUMN = "Описание задачи"
DEFAULT_REGION_COLUMN = "Регион"
DEFAULT_LABEL_COLUMN = "Тип"

_morph = None

def get_morph():
    global _morph
    if _morph is None:
        from pymorphy3 import MorphAnalyzer
        _morph = MorphAnalyzer()
    return _morph

@dataclass
class TrainingResult:
    best_model_name: str
    comparison_table: pd.DataFrame
    cv_results: dict[str, Any]
    pipeline: Pipeline

class TransformerEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper for SentenceTransformer."""
    def __init__(self, model_name: str = 'sentence-transformers/distiluse-base-multilingual-cased-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = get_sentence_transformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = get_sentence_transformer(self.model_name)
        # X is expected to be a list of strings or a Series
        if isinstance(X, pd.Series):
            X = X.tolist()
        return self.model.encode(X, show_progress_bar=False)

def build_parser(description: str = "Thesis Pipeline CLI") -> Any:
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", default="export_analytic_2026-03-12.csv", help="Dataset source path.")
    parser.add_argument("--model", default="best_transformer_model.pkl", help="Model save path.")
    parser.add_argument("--cv", type=int, default=5, help="Number of folds for Cross-Validation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold.")
    return parser

def mask_description_and_inn(text: object) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"(ИНН:\s*)(\d+)", r"\1[MASKED_INN]", text, flags=re.IGNORECASE)
    bm_pattern = r"(Бренд-менеджер:\s*)([А-Яа-яЁё]+)\s+([А-Яа-яЁё]+)"
    text = re.sub(bm_pattern, r"\1[ANONYMIZED_NAME]", text, flags=re.IGNORECASE)
    return text

def load_dataset(source: str) -> pd.DataFrame:
    """Load the ticket dataset with robust separator and encoding detection."""
    if Path(source).exists():
        # Common encodings for Cyrillic CSVs (Excel often uses windows-1251 or utf-8-sig)
        encodings = ["utf-8-sig", "utf-8", "windows-1251"]
        df = None
        
        for enc in encodings:
            try:
                # First try the custom '#' separator
                df = pd.read_csv(source, sep="#", quotechar='"', encoding=enc, engine="python")
                if len(df.columns) < 2:
                    raise ValueError("Not enough columns with # separator")
                break # Success
            except:
                try:
                    # Fallback to standard comma
                    df = pd.read_csv(source, encoding=enc)
                    if len(df.columns) >= 2:
                        break # Success
                except:
                    continue
        
        if df is None:
            raise ValueError(f"Could not parse {source} with any common encoding/separator combination.")
    else:
        # Remote or legacy path - default to UTF-8
        df = pd.read_csv(source, sep="#", quotechar='"', encoding="utf-8", engine="python")
    
    if "ID" in df.columns:
        df = df[~df["ID"].isin(range(1, 21))].copy()
    
    df = df.dropna(subset=[DEFAULT_TEXT_COLUMN, DEFAULT_LABEL_COLUMN]).copy()
    df[DEFAULT_REGION_COLUMN] = df[DEFAULT_REGION_COLUMN].fillna("не указан").astype(str)
    
    # Combined input for Transformer
    df['combined_input'] = "Регион: " + df[DEFAULT_REGION_COLUMN] + ". Описание: " + df[DEFAULT_TEXT_COLUMN].apply(mask_description_and_inn)
    
    return df

def save_artifacts(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

def load_artifacts(path: Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_ticket(pipeline: Pipeline, text: str, region: str, threshold: float = 0.25) -> tuple[str, float, str]:
    combined = f"Регион: {region}. Описание: {text}"
    # Input to the pipeline is a DataFrame for consistency if needed, 
    # but here we use the combined input directly for the Transformer step
    X = pd.Series([combined])
    
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)[0]
        max_idx = np.argmax(probas)
        max_proba = probas[max_idx]
        label = pipeline.classes_[max_idx]
        
        if max_proba >= threshold:
            return str(label), float(max_proba), "ml"
    
    # Fallback/Manual dummy for this demo
    return "manual_review", 0.0, "fallback"

def build_tfidf_pipeline(
    estimator: BaseEstimator,
    preprocess_func: Callable[[object], str],
    text_column: str = DEFAULT_TEXT_COLUMN,
    region_column: str = DEFAULT_REGION_COLUMN,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_func,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", vectorizer, text_column),
            ("region", OneHotEncoder(handle_unknown="ignore"), [region_column]),
        ],
        remainder="drop",
    )
    
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
