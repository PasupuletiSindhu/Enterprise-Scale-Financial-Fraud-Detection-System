"""
Comprehensive prediction module for fraud detection.
Combines single predictions, batch processing, and concept drift detection.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
# Concept drift detection (optional)
try:
    from river import drift
    from river import metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("Warning: River library not available. Drift detection will be disabled.")

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudPredictor:
    """Comprehensive fraud prediction system."""
    
    def __init__(self, models_dir: str = 'artifacts'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessor."""
        logger.info(f"Loading models from {self.models_dir}")
        
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Preprocessor loaded successfully")
            
            # Load models
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('.joblib') and not f.startswith('preprocessor') and not f.startswith('evaluation')]
            
            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        logger.info("Preprocessing input data...")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        df = self.preprocessor.engineer_features(df)
        
        # Encode categorical features
        df = self.preprocessor.encode_categorical_features(df, fit=False)
        
        # Scale features
        df = self.preprocessor.scale_features(df, fit=False)
        
        # Select only feature columns
        if hasattr(self.preprocessor, 'feature_columns'):
            df = df[self.preprocessor.feature_columns]
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def predict_fraud_probability(self, input_data: Dict[str, Any], 
                                model_name: str = 'hybrid_xgboost') -> Dict[str, Any]:
        """Predict fraud probability for a single transaction."""
        logger.info(f"Making prediction using {model_name}")
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Get model
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0, 1]
            else:
                # For models without predict_proba (like Isolation Forest)
                proba = model.predict(X)[0]
                if proba == -1:  # Anomaly
                    proba = 0.9
                else:
                    proba = 0.1
            
            # Create risk level
            if proba > 0.8:
                risk_level = "High"
            elif proba > 0.5:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            result = {
                'fraud_probability': float(proba),
                'risk_level': risk_level,
                'model_used': model_name,
                'transaction_id': input_data.get('transaction_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, input_data: List[Dict[str, Any]], 
                     model_name: str = 'hybrid_xgboost') -> List[Dict[str, Any]]:
        """Predict fraud probability for multiple transactions."""
        logger.info(f"Making batch predictions using {model_name}")
        
        results = []
        
        for i, transaction in enumerate(input_data):
            try:
                result = self.predict_fraud_probability(transaction, model_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting transaction {i}: {e}")
                results.append({
                    'fraud_probability': 0.0,
                    'risk_level': 'Error',
                    'model_used': model_name,
                    'transaction_id': transaction.get('transaction_id', f'tx_{i}'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Batch prediction completed: {len(results)} predictions")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'models_loaded': list(self.models.keys()),
            'preprocessor_loaded': self.preprocessor is not None,
            'models_directory': self.models_dir
        }
        
        if self.preprocessor:
            info['feature_columns'] = getattr(self.preprocessor, 'feature_columns', [])
            info['categorical_columns'] = getattr(self.preprocessor, 'categorical_columns', [])
        
        return info
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure."""
        logger.info("Validating input data...")
        
        # Check required fields
        required_fields = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check data types
        for field in required_fields:
            if not isinstance(input_data[field], (int, float)):
                raise ValueError(f"Field {field} must be numeric")
        
        # Check for negative amounts
        if input_data['Amount'] < 0:
            raise ValueError("Amount cannot be negative")
        
        logger.info("Input validation passed")
        return True

class ConceptDriftDetector:
    """Detect concept drift in fraud detection models."""
    
    def __init__(self):
        if RIVER_AVAILABLE:
            self.adwin_detector = drift.ADWIN()
            # Use PageHinkley instead of DDM (DDM doesn't exist in river)
            try:
                self.ddm_detector = drift.PageHinkley()
            except AttributeError:
                # Fallback to another drift detector
                self.ddm_detector = drift.ADWIN()
        else:
            self.adwin_detector = None
            self.ddm_detector = None
        self.drift_alerts = []
        
    def detect_drift(self, feature_values: List[float], 
                    predictions: List[float], 
                    actual_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Detect concept drift in feature distributions and model performance."""
        logger.info("Detecting concept drift...")
        
        drift_results = {
            'adwin_drift': False,
            'ddm_drift': False,
            'alerts': []
        }
        
        if not RIVER_AVAILABLE:
            logger.warning("River library not available. Drift detection disabled.")
            return drift_results
        
        # ADWIN drift detection on feature values
        for value in feature_values:
            self.adwin_detector.update(value)
            if self.adwin_detector.drift_detected:
                drift_results['adwin_drift'] = True
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'ADWIN',
                    'severity': 'High',
                    'message': 'Significant drift detected in feature distribution'
                }
                drift_results['alerts'].append(alert)
                self.drift_alerts.append(alert)
                break
        
        # PageHinkley drift detection on model performance (if labels available)
        if actual_labels is not None and len(actual_labels) > 0:
            metric = metrics.Accuracy()
            
            for pred, actual in zip(predictions, actual_labels):
                pred_label = 1 if pred > 0.5 else 0
                metric.update(actual, pred_label)
                
                self.ddm_detector.update(metric.get())
                if self.ddm_detector.drift_detected:
                    drift_results['ddm_drift'] = True
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'PageHinkley',
                        'severity': 'Medium',
                        'message': 'Model performance degradation detected'
                    }
                    drift_results['alerts'].append(alert)
                    self.drift_alerts.append(alert)
                    break
        
        logger.info(f"Drift detection completed. Alerts: {len(drift_results['alerts'])}")
        return drift_results
    
    def get_drift_alerts(self) -> List[Dict[str, Any]]:
        """Get all drift alerts."""
        return self.drift_alerts

class BatchProcessor:
    """Handle batch processing of fraud predictions."""
    
    def __init__(self, predictor: FraudPredictor):
        self.predictor = predictor
        self.drift_detector = ConceptDriftDetector()
        
    def process_batch(self, transactions: List[Dict[str, Any]], 
                     detect_drift: bool = True) -> Dict[str, Any]:
        """Process a batch of transactions with optional drift detection."""
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        # Make predictions
        predictions = self.predictor.predict_batch(transactions)
        
        # Extract features and probabilities for drift detection
        if detect_drift and len(transactions) > 0:
            feature_values = [tx.get('Amount', 0) for tx in transactions]
            pred_probabilities = [pred.get('fraud_probability', 0) for pred in predictions]
            
            # Detect drift
            drift_results = self.drift_detector.detect_drift(feature_values, pred_probabilities)
        else:
            drift_results = {'alerts': []}
        
        # Calculate batch statistics
        high_risk = sum(1 for p in predictions if p.get('risk_level') == 'High')
        medium_risk = sum(1 for p in predictions if p.get('risk_level') == 'Medium')
        low_risk = sum(1 for p in predictions if p.get('risk_level') == 'Low')
        
        batch_result = {
            'predictions': predictions,
            'total_transactions': len(transactions),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'fraud_rate': high_risk / len(transactions) if len(transactions) > 0 else 0,
            'drift_alerts': drift_results.get('alerts', []),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch processing completed. High risk: {high_risk}, Medium risk: {medium_risk}")
        return batch_result
    
    def get_drift_alerts(self) -> List[Dict[str, Any]]:
        """Get all drift alerts."""
        return self.drift_detector.get_drift_alerts()

def create_sample_transaction() -> Dict[str, Any]:
    """Create a sample transaction for testing."""
    return {
        'Time': 1000.0,
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 149.62,
        'hour_of_day': 14,
        'day_of_week': 2,
        'avg_amount_per_customer': 88.35,
        'tx_freq_24h': 1,
        'merchant_fraud_rate': 0.0017,
        'transaction_type': 'online',
        'card_type': 'credit',
        'region': 'US',
        'channel': 'web'
    }

def simulate_batch_inference(data_file: str, batch_size: int = 1000) -> Dict[str, Any]:
    """Simulate batch inference on historical data."""
    logger.info(f"Simulating batch inference on {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Initialize predictor and batch processor
    predictor = FraudPredictor()
    batch_processor = BatchProcessor(predictor)
    
    # Process in batches
    all_results = []
    drift_alerts = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        # Convert to list of dictionaries
        transactions = batch_df.to_dict('records')
        
        # Process batch
        batch_result = batch_processor.process_batch(transactions, detect_drift=True)
        all_results.append(batch_result)
        
        # Collect drift alerts
        drift_alerts.extend(batch_result['drift_alerts'])
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    
    # Aggregate results
    total_transactions = sum(r['total_transactions'] for r in all_results)
    total_high_risk = sum(r['high_risk_count'] for r in all_results)
    total_medium_risk = sum(r['medium_risk_count'] for r in all_results)
    total_low_risk = sum(r['low_risk_count'] for r in all_results)
    
    simulation_result = {
        'total_batches': len(all_results),
        'total_transactions': total_transactions,
        'total_high_risk': total_high_risk,
        'total_medium_risk': total_medium_risk,
        'total_low_risk': total_low_risk,
        'overall_fraud_rate': total_high_risk / total_transactions if total_transactions > 0 else 0,
        'drift_alerts': drift_alerts,
        'batch_results': all_results,
        'simulation_timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Batch inference simulation completed. Total transactions: {total_transactions}")
    return simulation_result

if __name__ == "__main__":
    # Example usage
    predictor = FraudPredictor()
    
    # Single prediction
    sample_tx = create_sample_transaction()
    result = predictor.predict_fraud_probability(sample_tx)
    print("Single prediction:", result)
    
    # Batch prediction
    batch_txs = [create_sample_transaction() for _ in range(5)]
    batch_results = predictor.predict_batch(batch_txs)
    print(f"Batch prediction: {len(batch_results)} results")
    
    # Model info
    info = predictor.get_model_info()
    print("Model info:", info) 