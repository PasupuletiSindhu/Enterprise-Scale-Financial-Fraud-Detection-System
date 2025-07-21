"""
Real-time Fraud Detection Processor
Handles streaming data for real-time fraud detection
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import queue
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeFraudProcessor:
    """Real-time fraud detection processor for streaming data."""
    
    def __init__(self, model_path: str = 'artifacts', 
                 preprocessor_path: str = 'artifacts/preprocessor.joblib',
                 buffer_size: int = 1000,
                 batch_size: int = 100,
                 update_interval: int = 60):
        """
        Initialize real-time processor.
        
        Args:
            model_path: Path to saved models
            preprocessor_path: Path to preprocessor
            buffer_size: Size of data buffer
            batch_size: Size of batches for processing
            update_interval: Interval for model updates (seconds)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        
        # Data buffers
        self.transaction_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.alert_buffer = deque(maxlen=100)
        
        # Processing queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Load models and preprocessor
        self.models = {}
        self.preprocessor = None
        self.load_models()
        
        # Performance tracking
        self.performance_metrics = {
            'total_processed': 0,
            'fraud_detected': 0,
            'avg_processing_time': 0,
            'last_update': datetime.now()
        }
        
        # Threading
        self.processing_thread = None
        self.is_running = False
        
        # Concept drift detection
        self.drift_detectors = {}
        self.drift_alerts = []
        
        logger.info("Real-time fraud processor initialized")
    
    def load_models(self):
        """Load trained models and preprocessor."""
        try:
            # Load preprocessor
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Preprocessor loaded successfully")
            
            # Load models
            model_files = {
                'xgboost': 'xgboost.joblib',
                'hybrid_xgboost': 'hybrid_xgboost.joblib'
            }
            
            # Try to load ensemble if it exists
            if os.path.exists(f"{self.model_path}/ensemble_xgboost.joblib"):
                model_files['ensemble_xgboost'] = 'ensemble_xgboost.joblib'
            
            for model_name, filename in model_files.items():
                try:
                    model_path = f"{self.model_path}/{filename}"
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load {model_name} model: {e}")
            
            # Load evaluation results for thresholds
            try:
                self.evaluation_results = joblib.load(f"{self.model_path}/evaluation_results.joblib")
                logger.info("Evaluation results loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load evaluation results: {e}")
                self.evaluation_results = {}
            
            # Load or create optimal thresholds
            try:
                self.optimal_thresholds = joblib.load(f"{self.model_path}/optimal_thresholds.joblib")
                logger.info("Model thresholds loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load thresholds: {e}")
                # Create default thresholds
                self.optimal_thresholds = {}
                for model_name in model_files.keys():
                    self.optimal_thresholds[model_name] = 0.5
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess a single transaction for prediction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            
            # Apply preprocessor
            if self.preprocessor:
                # Check if preprocessor has transform method
                if hasattr(self.preprocessor, 'transform'):
                    processed_df = self.preprocessor.transform(df)
                else:
                    # If it's a scaler, apply it directly
                    processed_df = self.preprocessor.fit_transform(df)
                return processed_df
            else:
                logger.warning("No preprocessor available")
                return df
                
        except Exception as e:
            logger.error(f"Error preprocessing transaction: {e}")
            return pd.DataFrame()
    
    def predict_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single transaction."""
        start_time = time.time()
        
        try:
            # Preprocess transaction
            processed_data = self.preprocess_transaction(transaction)
            
            if processed_data.empty:
                return {
                    'transaction_id': transaction.get('id', 'unknown'),
                    'fraud_probability': 0.0,
                    'prediction': 0,
                    'model_used': 'none',
                    'processing_time': time.time() - start_time,
                    'error': 'Preprocessing failed'
                }
            
            # Get predictions from all available models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(processed_data)[0, 1]
                        pred = (prob >= self.optimal_thresholds.get(model_name, 0.5)).astype(int)
                    else:
                        pred = model.predict(processed_data)[0]
                        prob = pred  # Use prediction as probability
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = prob
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name} prediction: {e}")
                    predictions[model_name] = 0
                    probabilities[model_name] = 0.0
            
            # Use ensemble prediction if available, otherwise use XGBoost
            if 'ensemble_xgboost' in predictions:
                final_prob = probabilities['ensemble_xgboost']
                final_pred = predictions['ensemble_xgboost']
                model_used = 'ensemble_xgboost'
            elif 'xgboost' in predictions:
                final_prob = probabilities['xgboost']
                final_pred = predictions['xgboost']
                model_used = 'xgboost'
            else:
                final_prob = 0.0
                final_pred = 0
                model_used = 'none'
            
            # Create result
            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'timestamp': datetime.now(),
                'fraud_probability': float(final_prob),
                'prediction': int(final_pred),
                'model_used': model_used,
                'all_predictions': predictions,
                'all_probabilities': probabilities,
                'processing_time': time.time() - start_time,
                'error': None
            }
            
            # Update performance metrics
            self.performance_metrics['total_processed'] += 1
            if final_pred == 1:
                self.performance_metrics['fraud_detected'] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics['avg_processing_time']
            new_time = result['processing_time']
            self.performance_metrics['avg_processing_time'] = (
                (current_avg * (self.performance_metrics['total_processed'] - 1) + new_time) / 
                self.performance_metrics['total_processed']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud prediction: {e}")
            return {
                'transaction_id': transaction.get('id', 'unknown'),
                'fraud_probability': 0.0,
                'prediction': 0,
                'model_used': 'none',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def process_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of transactions."""
        results = []
        
        for transaction in transactions:
            result = self.predict_fraud(transaction)
            results.append(result)
            
            # Add to buffers
            self.transaction_buffer.append(transaction)
            self.prediction_buffer.append(result)
            
            # Check for high-risk transactions
            if result['fraud_probability'] > 0.8:
                self.alert_buffer.append({
                    'timestamp': datetime.now(),
                    'transaction_id': result['transaction_id'],
                    'fraud_probability': result['fraud_probability'],
                    'severity': 'HIGH',
                    'message': f"High-risk transaction detected: {result['fraud_probability']:.4f}"
                })
        
        return results
    
    def start_processing(self):
        """Start the real-time processing thread."""
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop the real-time processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Real-time processing stopped")
    
    def _processing_loop(self):
        """Main processing loop for real-time data."""
        while self.is_running:
            try:
                # Process input queue
                batch = []
                while len(batch) < self.batch_size:
                    try:
                        transaction = self.input_queue.get_nowait()
                        batch.append(transaction)
                    except queue.Empty:
                        break
                
                if batch:
                    results = self.process_batch(batch)
                    
                    # Put results in output queue
                    for result in results:
                        self.output_queue.put(result)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def add_transaction(self, transaction: Dict[str, Any]):
        """Add a transaction to the processing queue."""
        self.input_queue.put(transaction)
    
    def get_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions from output queue."""
        predictions = []
        while len(predictions) < limit:
            try:
                prediction = self.output_queue.get_nowait()
                predictions.append(prediction)
            except queue.Empty:
                break
        return predictions
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get recent fraud alerts."""
        return list(self.alert_buffer)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['fraud_rate'] = (
            metrics['fraud_detected'] / metrics['total_processed'] * 100 
            if metrics['total_processed'] > 0 else 0
        )
        metrics['last_update'] = datetime.now()
        return metrics
    
    def detect_concept_drift(self, window_size: int = 1000) -> List[Dict[str, Any]]:
        """Detect concept drift in recent data."""
        if len(self.prediction_buffer) < window_size:
            return []
        
        # Get recent predictions
        recent_predictions = list(self.prediction_buffer)[-window_size:]
        
        # Calculate drift metrics
        fraud_rates = []
        avg_probabilities = []
        
        # Split into windows
        window_count = 10
        window_length = len(recent_predictions) // window_count
        
        for i in range(window_count):
            start_idx = i * window_length
            end_idx = start_idx + window_length
            window_data = recent_predictions[start_idx:end_idx]
            
            if window_data:
                fraud_rate = sum(1 for p in window_data if p['prediction'] == 1) / len(window_data)
                avg_prob = np.mean([p['fraud_probability'] for p in window_data])
                
                fraud_rates.append(fraud_rate)
                avg_probabilities.append(avg_prob)
        
        # Detect drift
        drift_alerts = []
        
        if len(fraud_rates) >= 2:
            # Check for significant changes in fraud rate
            fraud_rate_change = abs(fraud_rates[-1] - fraud_rates[0])
            if fraud_rate_change > 0.1:  # 10% change
                drift_alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'Fraud Rate Drift',
                    'severity': 'HIGH' if fraud_rate_change > 0.2 else 'MEDIUM',
                    'message': f"Fraud rate changed by {fraud_rate_change:.2%}",
                    'old_rate': fraud_rates[0],
                    'new_rate': fraud_rates[-1]
                })
        
        if len(avg_probabilities) >= 2:
            # Check for significant changes in average probability
            prob_change = abs(avg_probabilities[-1] - avg_probabilities[0])
            if prob_change > 0.1:  # 10% change
                drift_alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'Probability Drift',
                    'severity': 'HIGH' if prob_change > 0.2 else 'MEDIUM',
                    'message': f"Average probability changed by {prob_change:.2%}",
                    'old_prob': avg_probabilities[0],
                    'new_prob': avg_probabilities[-1]
                })
        
        return drift_alerts
    
    def generate_sample_transaction(self) -> Dict[str, Any]:
        """Generate a sample transaction for testing."""
        return {
            'id': f"txn_{int(time.time())}",
            'Time': np.random.randint(0, 100000),
            'Amount': np.random.uniform(1, 1000),
            'customer_id': f"cust_{np.random.randint(1, 1000)}",
            'merchant_id': f"merch_{np.random.randint(1, 1000)}",
            'transaction_type': np.random.choice(['online', 'in_store', 'atm']),
            'card_type': np.random.choice(['visa', 'mastercard', 'amex']),
            'region': np.random.choice(['US', 'EU', 'ASIA']),
            'channel': np.random.choice(['web', 'mobile', 'pos']),
            'V1': np.random.normal(0, 1),
            'V2': np.random.normal(0, 1),
            'V3': np.random.normal(0, 1),
            'V4': np.random.normal(0, 1),
            'V5': np.random.normal(0, 1),
            'V6': np.random.normal(0, 1),
            'V7': np.random.normal(0, 1),
            'V8': np.random.normal(0, 1),
            'V9': np.random.normal(0, 1),
            'V10': np.random.normal(0, 1),
            'V11': np.random.normal(0, 1),
            'V12': np.random.normal(0, 1),
            'V13': np.random.normal(0, 1),
            'V14': np.random.normal(0, 1),
            'V15': np.random.normal(0, 1),
            'V16': np.random.normal(0, 1),
            'V17': np.random.normal(0, 1),
            'V18': np.random.normal(0, 1),
            'V19': np.random.normal(0, 1),
            'V20': np.random.normal(0, 1),
            'V21': np.random.normal(0, 1),
            'V22': np.random.normal(0, 1),
            'V23': np.random.normal(0, 1),
            'V24': np.random.normal(0, 1),
            'V25': np.random.normal(0, 1),
            'V26': np.random.normal(0, 1),
            'V27': np.random.normal(0, 1),
            'V28': np.random.normal(0, 1)
        }

def main():
    """Test the real-time processor."""
    processor = RealTimeFraudProcessor()
    
    # Start processing
    processor.start_processing()
    
    # Generate and process sample transactions
    for i in range(10):
        transaction = processor.generate_sample_transaction()
        processor.add_transaction(transaction)
        time.sleep(0.5)
    
    # Get results
    predictions = processor.get_predictions()
    alerts = processor.get_alerts()
    metrics = processor.get_performance_metrics()
    
    print(f"Processed {len(predictions)} transactions")
    print(f"Generated {len(alerts)} alerts")
    print(f"Performance metrics: {metrics}")
    
    # Stop processing
    processor.stop_processing()

if __name__ == "__main__":
    main() 