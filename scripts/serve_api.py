"""
FastAPI server to serve model predictions
REST API server để serve model với giao diện web
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import ContractPredictor
from utils.logger import setup_logger

# Initialize
app = FastAPI(
    title="Contract Transformer API",
    description="API for Vietnamese contract classification using PhoBERT",
    version="1.0.0"
)

# Setup templates and static files
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor
predictor = None
logger = setup_logger('api')


# Request models
class PredictRequest(BaseModel):
    text: str
    return_probabilities: bool = False


class BatchPredictRequest(BaseModel):
    texts: List[str]
    return_probabilities: bool = False


class QAPredictRequest(BaseModel):
    context: str
    question: str


class NERRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50


# Response models
class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model khi start server"""
    global predictor
    model_path = os.getenv('MODEL_PATH', 'outputs/best_model.pt')
    
    try:
        logger.info(f"Loading model from {model_path}")
        predictor = ContractPredictor(model_path)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Web UI homepage"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", response_model=Dict)
async def api_info():
    """API info endpoint"""
    return {
        "message": "Contract Transformer API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "ner": "/ner",
            "qa": "/qa",
            "summarize": "/summarize",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict contract type from text
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predictor.predict(
            request.text,
            return_probs=request.return_probabilities
        )
        
        # Handle both old and new format
        label = result.get('label_name') or result.get('label') or str(result.get('predicted_label', ''))
        confidence = result.get('confidence', 0.0)
        
        response = {
            "label": label,
            "confidence": confidence
        }
        
        if request.return_probabilities:
            response['probabilities'] = result.get('probabilities', {})
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(request: BatchPredictRequest):
    """
    Batch prediction for multiple texts
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = predictor.predict_batch(
            request.texts,
            return_probabilities=request.return_probabilities
        )
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa")
async def qa(request: QAPredictRequest):
    """
    Question answering endpoint
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if predictor.task_type != 'qa':
        raise HTTPException(status_code=400, detail="Model is not trained for QA task")
    
    try:
        result = predictor.predict_qa(request.context, request.question)
        return result
    
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner")
async def ner(request: NERRequest):
    """
    Named Entity Recognition endpoint
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if predictor.task_type != 'ner':
        raise HTTPException(status_code=400, detail="Model is not trained for NER task")
    
    try:
        result = predictor.predict_ner(request.text)
        return result
    
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Text summarization endpoint
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if predictor.task_type != 'summarization':
        raise HTTPException(status_code=400, detail="Model is not trained for summarization task")
    
    try:
        result = predictor.predict_summary(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return result
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(
        "serve_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
