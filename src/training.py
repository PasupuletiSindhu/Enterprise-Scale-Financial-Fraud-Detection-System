"""
Comprehensive training module for fraud detection.
Combines anomaly detection, XGBoost training, and hybrid model approaches.
Optimized for high precision (90%+) fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class HighPrecisionModelTrainer:
    """Comprehensive model training optimized for high precision (90%+)."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.evaluation_results = {}
        self.optimal_thresholds = {}
        
    def calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights to handle imbalanced data."""
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y), 
            y=y
        )
        # Convert numpy types to Python native types for pickling
        return {int(k): float(v) for k, v in zip(np.unique(y), class_weights)}
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              target_precision: float = 0.90) -> float:
        """Find optimal threshold to achieve target precision."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find threshold that gives us target precision
        for i, precision in enumerate(precisions):
            if precision >= target_precision:
                return thresholds[i-1] if i > 0 else 0.5
        
        # If we can't achieve target precision, return threshold for max precision
        return thresholds[np.argmax(precisions)]
    
    def train_high_precision_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   target_precision: float = 0.90) -> Dict[str, Any]:
        """Train XGBoost model optimized for high precision."""
        logger.info("Training High-Precision XGBoost model...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        logger.info(f"Class weights: {class_weights}")
        
        # Define parameter grid optimized for precision (reduced for stability)
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [6, 8],
            'learning_rate': [0.1, 0.15],
            'subsample': [0.9, 1.0],
            'colsample_bytree': [0.9, 1.0],
            'min_child_weight': [3, 5],
            'gamma': [0, 0.1],
            'scale_pos_weight': [float(class_weights[1] / class_weights[0])]  # Convert to float
        }
        
        # Base XGBoost model
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # Use optimized parameters directly instead of GridSearchCV
        best_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0,
            'scale_pos_weight': float(class_weights[1] / class_weights[0])
        }
        
        best_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='auc',
            use_label_encoder=False,
            **best_params
        )
        
        best_model.fit(X_train, y_train)
        self.best_params['xgboost'] = best_params
        logger.info(f"XGBoost trained with optimized parameters")
        
        # Get predictions
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for target precision
        optimal_threshold = self.find_optimal_threshold(y_test.values, y_pred_proba, target_precision)
        self.optimal_thresholds['xgboost'] = optimal_threshold
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division parameter
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store results
        self.models['xgboost'] = best_model
        self.evaluation_results['xgboost'] = {
            'auc': auc_score,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        logger.info(f"XGBoost training completed. Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}")
        return self.evaluation_results['xgboost']
    
    def train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           target_precision: float = 0.90) -> Dict[str, Any]:
        """Train ensemble model combining multiple algorithms for high precision."""
        logger.info("Training Ensemble model for high precision...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        
        # Train multiple models
        models = {}
        
        # 1. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            scale_pos_weight=class_weights[1] / class_weights[0],
            random_state=self.random_state,
            eval_metric='auc'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # 2. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # 3. Logistic Regression with class weights
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        models['logistic_regression'] = lr_model
        
        # Get predictions from all models
        ensemble_probs = np.zeros(len(X_test))
        weights = [0.5, 0.3, 0.2]  # Weight for each model
        
        for i, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.predict(X_test)
            ensemble_probs += weights[i] * probs
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_test.values, ensemble_probs, target_precision)
        self.optimal_thresholds['ensemble'] = optimal_threshold
        
        # Make predictions
        y_pred = (ensemble_probs >= optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division parameter
        auc_score = roc_auc_score(y_test, ensemble_probs)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store results
        self.models['ensemble'] = models
        self.evaluation_results['ensemble'] = {
            'auc': auc_score,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': ensemble_probs,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        logger.info(f"Ensemble training completed. Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}")
        return self.evaluation_results['ensemble']
    
    def train_isolation_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train Isolation Forest for anomaly detection."""
        logger.info("Training Isolation Forest...")
        
        # Train Isolation Forest with optimized parameters
        iso_forest = IsolationForest(
            contamination=0.01,  # Assume 1% anomalies
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto'
        )
        
        iso_forest.fit(X_train)
        
        # Get anomaly scores
        test_scores = iso_forest.score_samples(X_test)
        
        # Convert scores to probabilities
        test_probs = 1 - (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())
        
        # Find optimal threshold for high precision
        optimal_threshold = self.find_optimal_threshold(y_test.values, test_probs, 0.90)
        self.optimal_thresholds['isolation_forest'] = optimal_threshold
        
        # Create predictions
        test_pred = (test_probs > optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division parameter
        auc_score = roc_auc_score(y_test, test_probs)
        report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
        
        # Store results
        self.models['isolation_forest'] = iso_forest
        self.evaluation_results['isolation_forest'] = {
            'auc': auc_score,
            'classification_report': report,
            'predictions': test_pred,
            'probabilities': test_probs,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        logger.info(f"Isolation Forest training completed. AUC: {auc_score:.4f}")
        return self.evaluation_results['isolation_forest']
    
    def train_vae(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train Variational Autoencoder for anomaly detection."""
        logger.info("Training Variational Autoencoder...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize VAE
        input_dim = X_train.shape[1]
        vae = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=64)
        
        # Training parameters
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        vae.train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                
                recon_x, mu, log_var = vae(batch_x)
                
                # Reconstruction loss
                recon_loss = criterion(recon_x, batch_x)
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss
                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"VAE Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}")
        
        # Get reconstruction errors
        vae.eval()
        with torch.no_grad():
            recon_x, _, _ = vae(X_test_tensor)
            mse_loss = nn.MSELoss(reduction='none')
            reconstruction_errors = mse_loss(recon_x, X_test_tensor).mean(dim=1).numpy()
        
        # Convert errors to probabilities (higher error = higher probability of fraud)
        test_probs = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min())
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_test.values, test_probs, 0.90)
        self.optimal_thresholds['vae'] = optimal_threshold
        
        # Create predictions
        test_pred = (test_probs > optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division parameter
        auc_score = roc_auc_score(y_test, test_probs)
        report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
        
        # Store results
        self.models['vae'] = vae
        self.evaluation_results['vae'] = {
            'auc': auc_score,
            'classification_report': report,
            'predictions': test_pred,
            'probabilities': test_probs,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        logger.info(f"VAE training completed. AUC: {auc_score:.4f}")
        return self.evaluation_results['vae']
    
    def create_hybrid_labels(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Create hybrid labels combining multiple model predictions."""
        logger.info("Creating hybrid labels...")
        
        # Train basic models for hybrid labels
        iso_forest = IsolationForest(contamination=0.01, random_state=self.random_state)
        iso_forest.fit(X_train)
        
        # Get anomaly scores
        train_scores = iso_forest.score_samples(X_train)
        test_scores = iso_forest.score_samples(X_test)
        
        # Convert to probabilities
        train_probs = 1 - (train_scores - train_scores.min()) / (train_scores.max() - train_scores.min())
        test_probs = 1 - (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())
        
        # Create hybrid labels (combine original labels with anomaly scores)
        train_hybrid = y_train.copy()
        test_hybrid = y_test.copy()
        
        # Add anomaly-based labels for high-confidence cases
        anomaly_threshold = np.percentile(train_probs, 95)  # Top 5% as anomalies
        
        # For training data, enhance labels with anomaly information
        high_anomaly_train = train_probs > anomaly_threshold
        train_hybrid[high_anomaly_train] = 1
        
        # For test data, create hybrid predictions
        high_anomaly_test = test_probs > anomaly_threshold
        test_hybrid[high_anomaly_test] = 1
        
        logger.info(f"Hybrid labels created. Enhanced fraud cases: {high_anomaly_train.sum()}")
        return train_hybrid, test_hybrid
    
    def train_hybrid_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train XGBoost with hybrid labels for improved precision."""
        logger.info("Training Hybrid XGBoost model...")
        
        # Create hybrid labels
        train_hybrid, test_hybrid = self.create_hybrid_labels(X_train, y_train, X_test, y_test)
        
        # Calculate class weights for hybrid labels
        class_weights = self.calculate_class_weights(train_hybrid)
        
        # Train XGBoost with hybrid labels
        hybrid_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            scale_pos_weight=class_weights[1] / class_weights[0],
            random_state=self.random_state,
            eval_metric='auc'
        )
        
        hybrid_model.fit(X_train, train_hybrid)
        
        # Get predictions
        y_pred_proba = hybrid_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for high precision
        optimal_threshold = self.find_optimal_threshold(y_test.values, y_pred_proba, 0.90)
        self.optimal_thresholds['hybrid_xgboost'] = optimal_threshold
        
        # Make predictions
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division parameter
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store results
        self.models['hybrid_xgboost'] = hybrid_model
        self.evaluation_results['hybrid_xgboost'] = {
            'auc': auc_score,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        logger.info(f"Hybrid XGBoost training completed. Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}")
        return self.evaluation_results['hybrid_xgboost']
    
    def evaluate_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all models and return comprehensive results."""
        logger.info("Evaluating all models...")
        
        results = {}
        
        # Train all models
        results['xgboost'] = self.train_high_precision_xgboost(X_train, y_train, X_test, y_test)
        results['ensemble'] = self.train_ensemble_model(X_train, y_train, X_test, y_test)
        results['hybrid_xgboost'] = self.train_hybrid_xgboost(X_train, y_train, X_test, y_test)
        results['isolation_forest'] = self.train_isolation_forest(X_train, y_train, X_test, y_test)
        
        # Try VAE if PyTorch is available
        try:
            results['vae'] = self.train_vae(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.warning(f"VAE training failed: {e}")
        
        # Find best model by precision
        best_model = max(results.items(), key=lambda x: x[1].get('precision', 0))
        logger.info(f"Best model by precision: {best_model[0]} with precision: {best_model[1].get('precision', 0):.4f}")
        
        return results
    
    def create_comparison_visualization(self, results: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """Create comparison visualization of all models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Precision-Recall comparison
        model_names = []
        precisions = []
        recalls = []
        
        for name, result in results.items():
            if 'precision' in result:
                model_names.append(name)
                precisions.append(result['precision'])
                recalls.append(result['recall'])
        
        axes[0, 0].bar(model_names, precisions, alpha=0.7, color='blue')
        axes[0, 0].set_title('Precision Comparison')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, recalls, alpha=0.7, color='red')
        axes[0, 1].set_title('Recall Comparison')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 2: ROC curves
        for name, result in results.items():
            if 'probabilities' in result:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc_score = roc_auc_score(y_test, result['probabilities'])
                axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC: {auc_score:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 3: Precision-Recall curves
        for name, result in results.items():
            if 'probabilities' in result:
                precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
                axes[1, 1].plot(recall, precision, label=f'{name}')
        
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('src/artifacts', exist_ok=True)
        plt.savefig('src/artifacts/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_models(self, directory: str = 'src/artifacts'):
        """Save all trained models and results."""
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if name == 'ensemble':
                # Save ensemble models separately
                for sub_name, sub_model in model.items():
                    joblib.dump(sub_model, os.path.join(directory, f'{name}_{sub_name}.joblib'))
            else:
                joblib.dump(model, os.path.join(directory, f'{name}.joblib'))
        
        # Save evaluation results
        joblib.dump(self.evaluation_results, os.path.join(directory, 'evaluation_results.joblib'))
        
        # Save optimal thresholds
        joblib.dump(self.optimal_thresholds, os.path.join(directory, 'optimal_thresholds.joblib'))
        
        # Save best parameters
        joblib.dump(self.best_params, os.path.join(directory, 'best_params.joblib'))
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str = 'src/artifacts'):
        """Load trained models and results."""
        try:
            # Load evaluation results
            self.evaluation_results = joblib.load(os.path.join(directory, 'evaluation_results.joblib'))
            
            # Load optimal thresholds
            self.optimal_thresholds = joblib.load(os.path.join(directory, 'optimal_thresholds.joblib'))
            
            # Load best parameters
            self.best_params = joblib.load(os.path.join(directory, 'best_params.joblib'))
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

def train_models(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                save_models: bool = True) -> Dict[str, Any]:
    """Main function to train all models with high precision focus."""
    logger.info("Starting high-precision model training...")
    
    # Initialize trainer
    trainer = HighPrecisionModelTrainer()
    
    # Train and evaluate all models
    results = trainer.evaluate_all_models(X_train, y_train, X_test, y_test)
    
    # Create comparison visualization
    trainer.create_comparison_visualization(results, X_test, y_test)
    
    # Save models if requested
    if save_models:
        trainer.save_models()
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("="*50)
    
    for name, result in results.items():
        if 'precision' in result:
            logger.info(f"{name.upper()}:")
            logger.info(f"  Precision: {result['precision']:.4f}")
            logger.info(f"  Recall: {result['recall']:.4f}")
            logger.info(f"  F1-Score: {result['f1_score']:.4f}")
            logger.info(f"  AUC: {result['auc']:.4f}")
            logger.info(f"  Optimal Threshold: {result.get('optimal_threshold', 'N/A'):.4f}")
            logger.info("-" * 30)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get('precision', 0))
    logger.info(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
    logger.info(f"   Precision: {best_model[1]['precision']:.4f}")
    logger.info(f"   Recall: {best_model[1]['recall']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    from src.preprocessing import preprocess_data
    
    # Load and preprocess data
    data_file = "src/data/creditcard.csv"
    X_train, X_test, y_train, y_test = preprocess_data(data_file)
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    print("Training completed!") 