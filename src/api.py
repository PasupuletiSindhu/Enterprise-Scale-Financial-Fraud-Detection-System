"""
Consolidated FastAPI for fraud detection with role-based access control.
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import secrets
from .prediction import FraudPredictor, create_sample_transaction, BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# User database (in production, use a proper database)
USERS_DB = {
    "fraud_analyst": {
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "fraud_analyst",
        "permissions": ["read", "predict", "batch_predict"]
    },
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "permissions": ["read", "predict", "batch_predict", "admin"]
    },
    "viewer": {
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "viewer",
        "permissions": ["read"]
    }
}

# API Keys (in production, use a proper key management system)
API_KEYS = {
    "fraud_analyst": "sk-fraud-analyst-123456789",
    "admin": "sk-admin-987654321",
    "viewer": "sk-viewer-456789123"
}

# Access logs
access_logs = []

class Transaction(BaseModel):
    """Transaction input model with validation."""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    avg_amount_per_customer: Optional[float] = None
    tx_freq_24h: Optional[int] = None
    merchant_fraud_rate: Optional[float] = None
    transaction_type: Optional[str] = "online"
    card_type: Optional[str] = "credit"
    region: Optional[str] = "US"
    channel: Optional[str] = "web"
    transaction_id: Optional[str] = None

    @validator('Amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount cannot be negative')
        return v

    @validator('Time')
    def validate_time(cls, v):
        if v < 0:
            raise ValueError('Time cannot be negative')
        return v

class BatchTransaction(BaseModel):
    """Batch transaction input model."""
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    """Prediction response model."""
    fraud_probability: float
    risk_level: str
    model_used: str
    transaction_id: Optional[str]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse]
    total_transactions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int

class AccessLog(BaseModel):
    """Access log model."""
    timestamp: str
    user: str
    role: str
    endpoint: str
    ip_address: str
    success: bool
    error_message: Optional[str] = None

def log_access(request: Request, user: str, role: str, endpoint: str, 
               success: bool, error_message: Optional[str] = None):
    """Log access events."""
    log_entry = AccessLog(
        timestamp=datetime.now().isoformat(),
        user=user,
        role=role,
        endpoint=endpoint,
        ip_address=request.client.host,
        success=success,
        error_message=error_message
    )
    
    access_logs.append(log_entry.dict())
    logger.info(f"Access: {user} ({role}) - {endpoint} - {'SUCCESS' if success else 'FAILED'}")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """Verify API key and return user info."""
    api_key = credentials.credentials
    
    for username, key in API_KEYS.items():
        if key == api_key:
            return {"username": username, "role": USERS_DB[username]["role"]}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(user_info: Dict[str, str] = Depends(verify_api_key)):
        username = user_info["username"]
        user_permissions = USERS_DB[username]["permissions"]
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return user_info
    return permission_checker

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Enterprise-scale fraud detection API with role-based access control",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Initialize predictor
try:
    predictor = FraudPredictor()
    batch_processor = BatchProcessor(predictor)
    logger.info("Fraud predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize fraud predictor: {e}")
    predictor = None
    batch_processor = None

@app.get("/")
def root():
    """Root endpoint - serve the dashboard."""
    return FileResponse(os.path.join("src", "static", "index.html"))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": predictor is not None
    }

@app.get("/info")
def get_api_info(user_info: Dict[str, str] = Depends(require_permission("read"))):
    """Get API information and model status."""
    log_access(
        request=Request,
        user=user_info["username"],
        role=user_info["role"],
        endpoint="/info",
        success=True
    )
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "api_version": "2.0.0",
        "models_loaded": predictor.get_model_info(),
        "user": user_info["username"],
        "role": user_info["role"],
        "permissions": USERS_DB[user_info["username"]]["permissions"]
    }

@app.post('/predict', response_model=PredictionResponse)
def predict_fraud(
    transaction: Transaction,
    user_info: Dict[str, str] = Depends(require_permission("predict"))
):
    """Predict fraud probability for a single transaction."""
    try:
        if predictor is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Validate input
        predictor.validate_input(transaction.dict())
        
        # Make prediction
        result = predictor.predict_fraud_probability(transaction.dict())
        
        # Log successful access
        log_access(
            request=Request,
            user=user_info["username"],
            role=user_info["role"],
            endpoint="/predict",
            success=True
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        log_access(
            request=Request,
            user=user_info["username"],
            role=user_info["role"],
            endpoint="/predict",
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_access(
            request=Request,
            user=user_info["username"],
            role=user_info["role"],
            endpoint="/predict",
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/batch-predict', response_model=BatchPredictionResponse)
def predict_fraud_batch(
    batch_data: BatchTransaction,
    user_info: Dict[str, str] = Depends(require_permission("batch_predict"))
):
    """Predict fraud probability for multiple transactions."""
    try:
        if batch_processor is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Convert to list of dictionaries
        transactions = [tx.dict() for tx in batch_data.transactions]
        
        # Process batch
        batch_result = batch_processor.process_batch(transactions)
        
        # Log successful access
        log_access(
            request=Request,
            user=user_info["username"],
            role=user_info["role"],
            endpoint="/batch-predict",
            success=True
        )
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in batch_result['predictions']],
            total_transactions=batch_result['total_transactions'],
            high_risk_count=batch_result['high_risk_count'],
            medium_risk_count=batch_result['medium_risk_count'],
            low_risk_count=batch_result['low_risk_count']
        )
        
    except Exception as e:
        log_access(
            request=Request,
            user=user_info["username"],
            role=user_info["role"],
            endpoint="/batch-predict",
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/sample-transaction')
def get_sample_transaction(user_info: Dict[str, str] = Depends(require_permission("read"))):
    """Get a sample transaction for testing."""
    log_access(
        request=Request,
        user=user_info["username"],
        role=user_info["role"],
        endpoint="/sample-transaction",
        success=True
    )
    
    return create_sample_transaction()

@app.get('/drift-alerts')
def get_drift_alerts(user_info: Dict[str, str] = Depends(require_permission("read"))):
    """Get concept drift alerts."""
    log_access(
        request=Request,
        user=user_info["username"],
        role=user_info["role"],
        endpoint="/drift-alerts",
        success=True
    )
    
    if batch_processor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    alerts = batch_processor.get_drift_alerts()
    return {
        "alerts": alerts,
        "total_alerts": len(alerts)
    }

@app.get('/access-logs')
def get_access_logs(user_info: Dict[str, str] = Depends(require_permission("admin"))):
    """Get access logs (admin only)."""
    log_access(
        request=Request,
        user=user_info["username"],
        role=user_info["role"],
        endpoint="/access-logs",
        success=True
    )
    
    return {
        "logs": access_logs[-100:],  # Return last 100 logs
        "total_logs": len(access_logs)
    }

@app.get('/api-keys')
def get_api_keys(user_info: Dict[str, str] = Depends(require_permission("admin"))):
    """Get API keys for users (admin only)."""
    log_access(
        request=Request,
        user=user_info["username"],
        role=user_info["role"],
        endpoint="/api-keys",
        success=True
    )
    
    return {
        "api_keys": {user: key for user, key in API_KEYS.items()},
        "users": {user: {"role": info["role"], "permissions": info["permissions"]} 
                 for user, info in USERS_DB.items()}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 