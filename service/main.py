from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import time
import re
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np

# Import our pipeline logic from the test folder
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "test"))
from thesis_pipeline import load_artifacts, DEFAULT_TEXT_COLUMN, DEFAULT_REGION_COLUMN

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, ROLES_DATA
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"CRITICAL: Model {MODEL_PATH} not found!")
    else:
        PIPELINE = load_artifacts(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")

    # 2. Load Roles
    if ROLES_PATH.exists():
        with open(ROLES_PATH, "r", encoding="utf-8") as f:
            ROLES_DATA = json.load(f)
    print(f"Roles loaded: {len(ROLES_DATA)} categories tracked.")
    
    yield

app = FastAPI(
    title="Ticket Categorization API",
    description="Microservice for automated ticket classification using TF-IDF and LinearSVC",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Templates (no more StaticFiles until we have a folder)
templates = Jinja2Templates(directory="service/templates")

# Configuration
MODEL_PATH = Path("test/best_transformer_model.pkl")
ROLES_PATH = Path("test/roles.json")
THRESHOLD = 0.2
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
# SLA Config
SLA_CONFIG = {
    "Заведение номенклатуры от менеджера": 1,
    "Запрос на обновление цен (сведенный поставщик)": 3,
    "Запрос на сведение поставщика": 14,
    "Запрос отчета": 14,
    "Изменение складской программы": 3,
    "Изменение/Добавление номенклатуры 1С8 ТХ": 7,
    "Расчет Спек закупки": 3
}

# Global Artifacts (Loaded once at startup)
PIPELINE = None
ROLES_DATA = {} # Store full roles for dynamic balancing
PROCESSED_TICKETS = [] # In-memory list for demo purposes
TICKET_COUNTER = 0

class TicketRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    text: str = Field(..., alias="Описание задачи")
    region: str = Field("не указан", alias="Регион")

class TicketResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: int
    text: str = Field(..., alias="Описание задачи")
    region: str = Field(..., alias="Регион")
    prediction: str
    confidence: float
    source: str
    responsible_worker: str
    completion_date: str
    latency_ms: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": PIPELINE is not None}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

@app.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: int):
    # Find the ticket in memory
    ticket = next((t for t in PROCESSED_TICKETS if t.get("id") == task_id), None)
    if not ticket:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return templates.TemplateResponse(
        request=request, 
        name="task_detail.html", 
        context={"ticket": ticket}
    )

@app.get("/tickets_list")
async def get_tickets():
    return PROCESSED_TICKETS

@app.post("/predict", response_model=Union[TicketResponse, List[TicketResponse]])
async def predict_ticket(request: Union[TicketRequest, List[TicketRequest]]):
    global PROCESSED_TICKETS, TICKET_COUNTER
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    is_batch = isinstance(request, list)
    requests = request if is_batch else [request]
    results = []

    for req in requests:
        start_time = time.perf_counter()
        TICKET_COUNTER += 1
        
        # Prepare input for pipeline
        X_input = pd.DataFrame([{
            DEFAULT_TEXT_COLUMN: req.text,
            DEFAULT_REGION_COLUMN: req.region
        }])

        # Cache pipeline attributes
        predict_proba = getattr(PIPELINE, "predict_proba", None)
        predict = getattr(PIPELINE, "predict", None)
        classes = getattr(PIPELINE, "classes_", None)

        # Inference logic optimization: using cached method references
        if predict_proba:
            probas = predict_proba(X_input)[0]
            max_idx = np.argmax(probas)
            max_proba = float(probas[max_idx])
            label = classes[max_idx]
            source = "ml"
            if max_proba < THRESHOLD:
                # Regex Fallback before final manual_review
                found_match = False
                for cat, pattern in REGEX_PATTERNS.items():
                    if re.search(pattern, req.text):
                        label = cat
                        source = "regex_fallback"
                        found_match = True
                        break
                
                if not found_match:
                    label = "manual_review"
                    source = "threshold_fallback"
        elif hasattr(PIPELINE, "decision_function"):
            scores = PIPELINE.decision_function(X_input)[0]
            # Convert distances to probabilities using softmax
            exp_scores = np.exp(scores - np.max(scores)) # shift for numerical stability
            probas = exp_scores / exp_scores.sum()
            
            max_idx = np.argmax(probas)
            max_proba = float(probas[max_idx])
            label = classes[max_idx]
            source = "ml"
            if max_proba < THRESHOLD:
                # Regex Fallback
                found_match = False
                for cat, pattern in REGEX_PATTERNS.items():
                    if re.search(pattern, req.text):
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

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Load balancing logic: pick worker with lowest current load
        responsible_worker = "Unassigned"
        if label in ROLES_DATA:
            workers = ROLES_DATA[label]
            # Find the worker string (key) with the minimum integer value
            best_id = min(workers, key=workers.get) 
            responsible_worker = best_id
            # Increment load for this worker in memory
            ROLES_DATA[label][best_id] += 1

        # Calculate Completion Date
        days_to_complete = SLA_CONFIG.get(label, 7)
        completion_dt = datetime.now() + timedelta(days=days_to_complete)
        completion_date_str = completion_dt.strftime("%d.%m.%Y")

        response_obj = TicketResponse(
            id=TICKET_COUNTER,
            text=req.text,
            region=req.region,
            prediction=label,
            responsible_worker=responsible_worker,
            completion_date=completion_date_str,
            confidence=max_proba,
            source=source,
            latency_ms=round(latency_ms, 4)
        )
        
        # Store in-memory for the dashboard
        PROCESSED_TICKETS.append(response_obj.dict())
        results.append(response_obj)

    return results if is_batch else results[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
