import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, classification_report
import time
import re

# START GLOBAL TIMING
GLOBAL_START_TIME = time.perf_counter()

import json
import numpy as np
from pathlib import Path
from thesis_pipeline import (
    load_dataset, 
    load_artifacts,
    DEFAULT_TEXT_COLUMN,
    DEFAULT_REGION_COLUMN,
    DEFAULT_LABEL_COLUMN,
    build_parser
)

def evaluate_production_model(args):
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {args.model} not found. Please run train_thesis_model.py first.")
        return

    print(f"Loading dataset from {args.source}...")
    df = load_dataset(args.source)
    print(f"Loading pipeline from {args.model}...")
    pipeline = load_artifacts(model_path)
    
    # Regex Fallback Patterns
    REGEX_PATTERNS = {
        "Заведение номенклатуры от менеджера": r"(?i)завест|создат|номенкл|карточк",
        "Запрос на обновление цен (сведенный поставщик)": r"(?i)обнов|цен|прайс|измен.*стоим",
        "Запрос на сведение поставщика": r"(?i)свест|сведен|нов.*постав",
        "Запрос отчета": r"(?i)отчет|выгрузк",
        "Изменение складской программы": r"(?i)склад|программ|остат",
        "Изменение/Добавление номенклатуры 1С8 ТХ": r"(?i)1с|тх|характер",
        "Расчет Спек закупки": r"(?i)спек|закуп|расчет"
    }

    # Identify labels and columns
    y_true = df[DEFAULT_LABEL_COLUMN].astype(str).tolist()

    y_pred = []
    latencies = []
    sources = []
    confidences = []
    
    print(f"Evaluating Pipeline on {len(df)} items using hybrid logic...")
    
    for _, row in df.iterrows():
        text = str(row.get(DEFAULT_TEXT_COLUMN, ""))
        region = str(row.get(DEFAULT_REGION_COLUMN, "не указан"))
        
        start = time.perf_counter()
        
        X_input = pd.DataFrame([{DEFAULT_TEXT_COLUMN: text, DEFAULT_REGION_COLUMN: region}])

        # Cache pipeline attributes
        predict_proba = getattr(pipeline, "predict_proba", None)
        predict = getattr(pipeline, "predict", None)
        classes = getattr(pipeline, "classes_", None)

        # Hybrid Inference Logic
        if predict_proba:
            probas = predict_proba(X_input)[0]
            max_idx = np.argmax(probas)
            max_proba = float(probas[max_idx])
            label = classes[max_idx]
            source = "ml"
            if max_proba < args.threshold:
                # Regex Fallback
                found_match = False
                for cat, pattern in REGEX_PATTERNS.items():
                    if re.search(pattern, text):
                        label = cat
                        source = "regex_fallback"
                        found_match = True
                        break
                if not found_match:
                    label = "manual_review"
                    source = "threshold_fallback"
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_input)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probas = exp_scores / exp_scores.sum()
            max_idx = np.argmax(probas)
            max_proba = float(probas[max_idx])
            label = classes[max_idx]
            source = "ml"
            if max_proba < args.threshold:
                # Regex Fallback
                found_match = False
                for cat, pattern in REGEX_PATTERNS.items():
                    if re.search(pattern, text):
                        label = cat
                        source = "regex_fallback"
                        found_match = True
                        break
                if not found_match:
                    label = "manual_review"
                    source = "threshold_fallback"
        else:
            label = predict(X_input)[0]
            max_proba = 1.0
            source = "ml"
            
        elapsed = time.perf_counter() - start
        
        y_pred.append(str(label))
        sources.append(source)
        confidences.append(float(max_proba))
        latencies.append(elapsed)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    # TOTAL WALL TIME
    total_process_time = time.perf_counter() - GLOBAL_START_TIME

    # Decision Source Distribution
    source_counts = pd.Series(sources).value_counts(normalize=True).to_dict()

    # Save results
    results_path = "test_evaluation_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== HYBRID PIPELINE EVALUATION (evaluate_production.py) ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy:          {acc:.6f}\n")
        f.write(f"Balanced Acc:      {bal_acc:.6f}\n")
        f.write(f"F1 Weighted:       {f1:.6f}\n")
        f.write(f"Avg Latency (Inf): {avg_latency:.6f} sec\n")
        f.write(f"Max Latency (Inf): {max_latency:.6f} sec\n")
        f.write(f"Total Wall Time:   {total_process_time:.6f} sec\n\n")
        
        f.write("Decision Source Distribution:\n")
        for src, val in source_counts.items():
            f.write(f"{src}: {val*100:.2f}%\n")
        f.write("\n")
        
        f.write("Detailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))

    # Confusion Matrix Plot
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=unique_labels, 
                yticklabels=unique_labels)
    plt.title(f"Confusion Matrix: {Path(args.model).name}")
    plt.ylabel("Actual Category")
    plt.xlabel("Predicted Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("test/test_confusion_matrix.png")
    plt.close()

    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to: {results_path}")
    print(f"Matrix saved to: test/test_confusion_matrix.png")
    print(f"Final Weighted F1: {f1:.4f}")

if __name__ == "__main__":
    parser = build_parser(description="Evaluate the hybrid pipeline.")
    parser.set_defaults(model="test/best_transformer_model.pkl")
    args = parser.parse_args()
    evaluate_production_model(args)
