# /train_thesis_model.py
"""Train the best ticket categorization model using Transformers and 5-Fold CV."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Core tools
from thesis_pipeline import (
    load_dataset, 
    save_artifacts, 
    build_parser,
    TransformerEmbeddingExtractor,
    DEFAULT_LABEL_COLUMN,
    preprocess_cleaned_cyr,
    preprocess_raw,
    build_tfidf_pipeline
)

def build_candidate_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    extractor = TransformerEmbeddingExtractor()
    return {
        "Transformer (Raw) + LogReg": Pipeline([
            ("extractor", extractor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
        ]),
        "Transformer (Raw) + LinearSVC": Pipeline([
            ("extractor", extractor),
            ("clf", LinearSVC(dual=False, class_weight="balanced", random_state=random_state))
        ]),
        "TF-IDF Cleaned(Cyr) + LinearSVC": build_tfidf_pipeline(
            LinearSVC(dual=False, class_weight="balanced", random_state=random_state),
            preprocess_func=preprocess_cleaned_cyr
        ),
        "TF-IDF Cleaned(Cyr) + LogReg": build_tfidf_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            preprocess_func=preprocess_cleaned_cyr
        ),
        "TF-IDF Raw + RandomForest": build_tfidf_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight="balanced"),
            preprocess_func=preprocess_raw
        ),
    }

def main() -> None:
    args = build_parser(description="Experimental: Train with Transformer + CV.").parse_args()
    
    print(f"Loading dataset from: {args.source}...")
    df = load_dataset(args.source)
    y = df[DEFAULT_LABEL_COLUMN]
    
    # Pre-calculate embeddings ONCE to save 3x the time
    print("\n[STEP 1] Generating Transformer Embeddings once for all models...")
    extractor = TransformerEmbeddingExtractor()
    X_embeddings = extractor.fit_transform(df['combined_input'])
    print(f"Embeddings generated. Shape: {X_embeddings.shape}")

    # Build simple models to run on top of pre-calculated embeddings
    embedding_models = {
        "Transformer (Raw) + LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.random_state),
        "Transformer (Raw) + LinearSVC": LinearSVC(dual=False, class_weight="balanced", random_state=args.random_state),
    }

    # Standard TF-IDF models still need the full DataFrame
    tfidf_pipelines = {
        "TF-IDF Cleaned(Cyr) + LinearSVC": build_tfidf_pipeline(
            LinearSVC(dual=False, class_weight="balanced", random_state=args.random_state),
            preprocess_func=preprocess_cleaned_cyr
        ),
        "TF-IDF Cleaned(Cyr) + LogReg": build_tfidf_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.random_state),
            preprocess_func=preprocess_cleaned_cyr
        ),
        "TF-IDF Raw + RandomForest": build_tfidf_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=args.random_state, n_jobs=-1, class_weight="balanced"),
            preprocess_func=preprocess_raw
        ),
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    results = []
    
    print(f"\n[STEP 2] Evaluating models using {args.cv}-Fold Cross-Validation...")

    # Evaluate Transformer models using pre-calculated X_embeddings
    for name, model in embedding_models.items():
        print(f" - Running CV for: {name} (using pre-cached embeddings)")
        scores = cross_val_score(model, X_embeddings, y, cv=cv, scoring="f1_weighted")
        results.append({
            "Model": name,
            "Mean F1 (Weighted)": np.mean(scores),
            "Std Dev": np.std(scores)
        })

    # Evaluate TF-IDF models using the DataFrame
    for name, pipeline in tfidf_pipelines.items():
        print(f" - Running CV for: {name} (standard pipeline)")
        scores = cross_val_score(pipeline, df, y, cv=cv, scoring="f1_weighted")
        results.append({
            "Model": name,
            "Mean F1 (Weighted)": np.mean(scores),
            "Std Dev": np.std(scores)
        })

    comparison_df = pd.DataFrame(results).sort_values(by="Mean F1 (Weighted)", ascending=False)
    print("\n" + "#" * 50)
    print("      CROSS-VALIDATION COMPARISON RESULTS")
    print("#" * 50)
    print(comparison_df.to_string(index=False))
    
    # Identify the overall best
    best_row = comparison_df.iloc[0]
    best_name = best_row["Model"]
    best_mean_f1 = best_row["Mean F1 (Weighted)"]
    print(f"\nBest Model: {best_name} (F1: {best_mean_f1:.4f})")
    
    # Final Refit on the ENTIRE dataset to save for production
    print(f"Final training of {best_name} on the full dataset...")
    
    if "Transformer" in best_name:
        # Re-build a production pipeline that includes the extractor
        final_model = embedding_models[best_name]
        full_pipeline = Pipeline([
            ("extractor", extractor),
            ("clf", final_model)
        ])
        full_pipeline.fit(df['combined_input'], y)
    else:
        full_pipeline = tfidf_pipelines[best_name]
        full_pipeline.fit(df, y)
    
    output_path = Path(args.model)
    save_artifacts(full_pipeline, output_path)
    print(f"Saved artifacts to {output_path}")

if __name__ == "__main__":
    main()
