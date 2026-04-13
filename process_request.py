# /process_request.py
"""Simulate production request with TF-IDF Hybrid Logic."""
from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
from thesis_pipeline import load_artifacts, DEFAULT_TEXT_COLUMN, DEFAULT_REGION_COLUMN

def main() -> None:
    parser = argparse.ArgumentParser(description="Process a ticket request from JSON.")
    parser.add_argument("input", help="Input JSON file path.")
    parser.add_argument("output", help="Output JSON file path.")
    parser.add_argument("--model", default="best_transformer_model.pkl", help="Model path.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold.")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {args.model} not found.")
        return

    # Load artifacts (TF-IDF + LinearSVC)
    pipeline = load_artifacts(model_path)

    # Load roles for responsible worker logic
    roles_path = Path("roles.json")
    roles_data = {}
    if roles_path.exists():
        with open(roles_path, "r", encoding="utf-8") as f:
            roles_data = json.load(f)
    
    # Pre-calculate first available worker for every category to avoid dict lookups in the loop
    worker_map = {cat: list(workers.keys())[0] for cat, workers in roles_data.items() if workers}

    # SLA logic (time to complete in days) 
    sla_config = {
        "Заведение номенклатуры от менеджера": 1,
        "Запрос на обновление цен (сведенный поставщик)": 3,
        "Запрос на сведение поставщика": 14,
        "Запрос отчета": 14,
        "Изменение складской программы": 3,
        "Изменение/Добавление номенклатуры 1С8 ТХ": 7,
        "Расчет Спек закупки": 3
    }

    # Read input JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = data if isinstance(data, list) else [data]
    results = []

    import pandas as pd
    import numpy as np

    # Cache pipeline attributes
    predict_proba = getattr(pipeline, "predict_proba", None)
    predict = getattr(pipeline, "predict", None)
    classes = getattr(pipeline, "classes_", None)

    for item in items:
        text = item.get("Описание задачи", "")
        region = item.get("Регион", "не указан")
        
        X_input = pd.DataFrame([{DEFAULT_TEXT_COLUMN: text, DEFAULT_REGION_COLUMN: region}])

        # using cached method references
        if predict_proba:
            probas = predict_proba(X_input)[0]
            max_idx = np.argmax(probas)
            max_proba = probas[max_idx]
            label = classes[max_idx]
            source = "ml"
            if max_proba < args.threshold:
                label = "manual_review"
                source = "threshold_fallback"
        else:
            label = predict(X_input)[0]
            max_proba = 1.0
            source = "ml"
        

        # O(1) Lookups 
        item.update({
            "prediction": label,
            "confidence": float(max_proba),
            "source": source,
            "responsible_worker": worker_map.get(label, "Unassigned"),
            "time_to_complete_days": sla_config.get(label, 7)
        })
        results.append(item)

    # Save output
    output_data = results if isinstance(data, list) else results[0]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Processed {len(results)} items. Saved to {args.output}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed_time:.4f} seconds")
