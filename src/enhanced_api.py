"""
Enhanced FastAPI backend with WebSocket support for real-time fraud detection dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import prediction module with proper path handling
try:
    from src.prediction import FraudPredictor, create_sample_transaction
except ImportError:
    # Try relative import if absolute import fails
    try:
        from .prediction import FraudPredictor, create_sample_transaction
    except ImportError:
        logger.error("Could not import FraudPredictor. Make sure the module is in the Python path.")
        # Create a mock FraudPredictor for demonstration
        class FraudPredictor:
            def predict_single(self, transaction_data):
                return {
                    "fraud_probability": np.random.beta(2, 8),
                    "risk_level": "Low" if np.random.random() > 0.1 else "High",
                    "model_used": "mock_model"
                }
            
        def create_sample_transaction():
            return {
                "Time": np.random.randint(0, 86400),
                "Amount": np.random.exponential(50),
                "V1": np.random.normal(0, 1),
                # Add other V features
                "transaction_id": f"tx_{np.random.randint(10000)}"
            }

# Security
security = HTTPBearer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.fraud_predictor = None
        self.last_update = datetime.now()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if self.active_connections:
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")
                    # Remove broken connection
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
                        
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

# Initialize connection manager
manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="Enterprise Fraud Detection API",
    description="Real-time fraud detection system with WebSocket support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication function
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification - replace with proper JWT validation."""
    token = credentials.credentials
    # In production, validate JWT token here
    if token != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Real-time data generation
class RealTimeDataGenerator:
    def __init__(self):
        self.transaction_id = 0
        self.fraud_count = 0
        self.total_transactions = 0
        
    def generate_transaction(self) -> Dict[str, Any]:
        """Generate realistic transaction data."""
        self.transaction_id += 1
        self.total_transactions += 1
        
        # Simulate realistic transaction patterns
        amount = np.random.exponential(50) + 10
        hour = np.random.randint(0, 24)
        
        # Higher fraud probability during certain hours
        fraud_prob = 0.003  # Base fraud rate
        if hour in [2, 3, 4]:  # Late night
            fraud_prob *= 3
        elif hour in [9, 10, 11]:  # Morning rush
            fraud_prob *= 1.5
            
        is_fraud = np.random.random() < fraud_prob
        if is_fraud:
            self.fraud_count += 1
            
        return {
            "transaction_id": self.transaction_id,
            "timestamp": datetime.now().isoformat(),
            "amount": round(amount, 2),
            "customer_id": np.random.randint(1, 10001),
            "merchant_id": np.random.randint(1, 101),
            "is_fraud": is_fraud,
            "fraud_probability": np.random.beta(2, 8) if not is_fraud else np.random.beta(8, 2),
            "hour_of_day": hour,
            "day_of_week": np.random.randint(0, 7),
            "region": np.random.choice(["US", "EU", "ASIA", "LATAM"]),
            "channel": np.random.choice(["web", "mobile", "pos"]),
            "card_type": np.random.choice(["credit", "debit", "prepaid"])
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        fraud_rate = (self.fraud_count / self.total_transactions * 100) if self.total_transactions > 0 else 0
        return {
            "total_transactions": self.total_transactions,
            "fraud_count": self.fraud_count,
            "fraud_rate": round(fraud_rate, 3),
            "timestamp": datetime.now().isoformat()
        }

# Initialize data generator
data_generator = RealTimeDataGenerator()

# WebSocket endpoint for real-time dashboard
@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial data
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to fraud detection dashboard",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        # Start real-time data stream
        while True:
            # Generate new transaction
            transaction = data_generator.generate_transaction()
            
            # Get current statistics
            stats = data_generator.get_statistics()
            
            # Prepare real-time update
            update_data = {
                "type": "real_time_update",
                "transaction": transaction,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            await manager.broadcast(update_data)
            
            # Wait before next update
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# REST API endpoints
@app.get("/")
async def root():
    """Root endpoint with dashboard info."""
    return {
        "message": "Enterprise Fraud Detection API",
        "version": "2.0.0",
        "status": "active",
        "websocket_endpoint": "/ws/dashboard",
        "documentation": "/docs"
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get current fraud detection statistics."""
    return data_generator.get_statistics()

@app.get("/api/transactions/recent")
async def get_recent_transactions(limit: int = 50):
    """Get recent transactions."""
    # Generate recent transactions
    transactions = []
    for _ in range(limit):
        transactions.append(data_generator.generate_transaction())
    return {"transactions": transactions}

@app.post("/api/predict")
async def predict_fraud(transaction_data: Dict[str, Any]):
    """Predict fraud for a single transaction."""
    try:
        # Initialize predictor if not already done
        if manager.fraud_predictor is None:
            try:
                manager.fraud_predictor = FraudPredictor()
            except Exception as e:
                logger.error(f"Could not initialize FraudPredictor: {e}")
                # Return mock prediction
                return {
                    "transaction_id": transaction_data.get("transaction_id", "unknown"),
                    "prediction": {
                        "fraud_probability": np.random.beta(2, 8),
                        "risk_level": "Low" if np.random.random() > 0.1 else "High",
                        "model_used": "mock_model"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
        # Make prediction
        prediction = manager.fraud_predictor.predict_fraud_probability(transaction_data)
        
        return {
            "transaction_id": transaction_data.get("transaction_id", "unknown"),
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts():
    """Get current system alerts."""
    alerts = []
    
    # Check for high fraud rate
    stats = data_generator.get_statistics()
    if stats["fraud_rate"] > 1.0:
        alerts.append({
            "type": "high_fraud_rate",
            "severity": "high",
            "message": f"Fraud rate is {stats['fraud_rate']}% - above threshold",
            "timestamp": datetime.now().isoformat()
        })
    
    # Check for system performance
    if len(manager.active_connections) > 100:
        alerts.append({
            "type": "high_connections",
            "severity": "medium",
            "message": f"High number of connections: {len(manager.active_connections)}",
            "timestamp": datetime.now().isoformat()
        })
        
    return {"alerts": alerts}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "uptime": str(datetime.now() - manager.last_update),
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Enhanced API starting up")
    manager.last_update = datetime.now()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Enhanced API shutting down")

# If running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 