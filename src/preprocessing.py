"""
Comprehensive preprocessing module for fraud detection.
Combines data loading, validation, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Comprehensive data preprocessing and feature engineering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = ['transaction_type', 'card_type', 'region', 'channel']
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality and structure."""
        logger.info("Validating data...")
        
        # Check for required columns
        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
        
        # Check data types
        logger.info("Data validation completed successfully")
        return True
    
    def generate_enterprise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enterprise-scale features for fraud detection."""
        logger.info("Generating enterprise features...")
        
        df = df.copy()
        
        # Generate customer IDs (simulate ~100,000 unique customers)
        np.random.seed(42)
        n_customers = 100000
        customer_weights = np.random.exponential(scale=1000, size=n_customers)
        customer_weights = customer_weights / customer_weights.sum()
        
        df['customer_id'] = np.random.choice(
            range(1, n_customers + 1), 
            size=len(df), 
            p=customer_weights
        )
        
        # Generate merchant IDs (simulate ~50,000 unique merchants)
        n_merchants = 50000
        merchant_weights = np.random.exponential(scale=500, size=n_merchants)
        merchant_weights = merchant_weights / merchant_weights.sum()
        
        df['merchant_id'] = np.random.choice(
            range(1, n_merchants + 1), 
            size=len(df), 
            p=merchant_weights
        )
        
        # Expand time to simulate a full month
        time_span = 32 * 24 * 60 * 60  # 32 days in seconds
        df['Time'] = np.random.uniform(0, time_span, len(df))
        
        # Add categorical features
        transaction_types = ['online', 'in_store', 'atm', 'mobile']
        card_types = ['credit', 'debit', 'prepaid']
        regions = ['US', 'EU', 'ASIA', 'LATAM', 'AFRICA']
        channels = ['web', 'mobile', 'pos', 'atm']
        
        df['transaction_type'] = np.random.choice(transaction_types, len(df))
        df['card_type'] = np.random.choice(card_types, len(df))
        df['region'] = np.random.choice(regions, len(df))
        df['channel'] = np.random.choice(channels, len(df))
        
        # Preserve fraud distribution
        fraud_indices = df[df['Class'] == 1].index
        df.loc[fraud_indices, 'customer_id'] = np.random.choice(
            range(1, n_customers + 1), 
            size=len(fraud_indices)
        )
        df.loc[fraud_indices, 'merchant_id'] = np.random.choice(
            range(1, n_merchants + 1), 
            size=len(fraud_indices)
        )
        
        logger.info(f"Enterprise features generated. Customers: {df['customer_id'].nunique()}, Merchants: {df['merchant_id'].nunique()}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data."""
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Time-based features
        df['hour_of_day'] = (df['Time'] % 24).astype(int)
        df['day_of_week'] = ((df['Time'] // 24) % 7).astype(int)
        
        # Customer-level features
        if 'customer_id' in df.columns:
            customer_stats = df.groupby('customer_id').agg({
                'Amount': 'mean',
                'Time': 'count'
            }).reset_index()
            customer_stats.columns = ['customer_id', 'avg_amount_per_customer', 'tx_freq_24h']
            df = df.merge(customer_stats, on='customer_id', how='left')
        else:
            # Fallback for datasets without customer_id
            df['avg_amount_per_customer'] = df['Amount'].mean()
            df['tx_freq_24h'] = 1
        
        # Merchant-level features
        if 'merchant_id' in df.columns:
            merchant_stats = df.groupby('merchant_id')['Class'].mean().reset_index()
            merchant_stats.columns = ['merchant_id', 'merchant_fraud_rate']
            df = df.merge(merchant_stats, on='merchant_id', how='left')
        else:
            df['merchant_fraud_rate'] = df['Class'].mean()
        
        # Fill missing values
        df['merchant_fraud_rate'] = df['merchant_fraud_rate'].fillna(0)
        df['avg_amount_per_customer'] = df['avg_amount_per_customer'].fillna(df['Amount'].mean())
        df['tx_freq_24h'] = df['tx_freq_24h'].fillna(1)
        
        logger.info("Feature engineering completed")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        logger.info("Encoding categorical features...")
        
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info("Categorical encoding completed")
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        logger.info("Scaling features...")
        
        # Select numerical features (exclude target and categorical)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['Class'] + self.categorical_columns]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        self.feature_columns = numerical_cols
        logger.info(f"Feature scaling completed. Features: {len(numerical_cols)}")
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training by separating features and target."""
        logger.info("Preparing data for training...")
        
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor artifacts."""
        logger.info(f"Saving preprocessor to {filepath}")
        
        artifacts = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns
        }
        
        joblib.dump(artifacts, filepath)
        logger.info("Preprocessor saved successfully")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor artifacts."""
        logger.info(f"Loading preprocessor from {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        artifacts = joblib.load(filepath)
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.feature_columns = artifacts['feature_columns']
        self.categorical_columns = artifacts['categorical_columns']
        
        logger.info("Preprocessor loaded successfully")

def analyze_dataset(filepath: str) -> Dict[str, Any]:
    """Analyze the fraud dataset and generate insights."""
    logger.info("Analyzing fraud dataset...")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic statistics
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'class_distribution': df['Class'].value_counts().to_dict(),
        'fraud_rate': (df['Class'].sum() / len(df)) * 100,
        'amount_stats': {
            'mean': df['Amount'].mean(),
            'std': df['Amount'].std(),
            'min': df['Amount'].min(),
            'max': df['Amount'].max(),
            'median': df['Amount'].median()
        }
    }
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Class distribution
    plt.subplot(2, 3, 1)
    df['Class'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Amount distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['Amount'], bins=50, alpha=0.7)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    
    # Amount by class
    plt.subplot(2, 3, 3)
    df.boxplot(column='Amount', by='Class')
    plt.title('Amount Distribution by Class')
    plt.suptitle('')
    
    # Time distribution
    plt.subplot(2, 3, 4)
    plt.hist(df['Time'], bins=50, alpha=0.7)
    plt.title('Transaction Time Distribution')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # V1 distribution
    plt.subplot(2, 3, 5)
    plt.hist(df['V1'], bins=50, alpha=0.7)
    plt.title('V1 Feature Distribution')
    plt.xlabel('V1')
    plt.ylabel('Frequency')
    
    # Correlation heatmap (sample features)
    plt.subplot(2, 3, 6)
    sample_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
    correlation_matrix = df[sample_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Dataset analysis completed")
    return analysis

def check_real_world_characteristics(filepath: str) -> Dict[str, Any]:
    """Check if the dataset has real-world characteristics."""
    logger.info("Checking real-world characteristics...")
    
    df = pd.read_csv(filepath)
    
    # Generate enterprise features
    preprocessor = DataPreprocessor()
    df_enhanced = preprocessor.generate_enterprise_features(df)
    df_enhanced = preprocessor.engineer_features(df_enhanced)
    
    # Analyze distributions
    characteristics = {
        'customer_distribution': {
            'total_customers': df_enhanced['customer_id'].nunique(),
            'transactions_per_customer': df_enhanced.groupby('customer_id').size().describe().to_dict()
        },
        'merchant_distribution': {
            'total_merchants': df_enhanced['merchant_id'].nunique(),
            'transactions_per_merchant': df_enhanced.groupby('merchant_id').size().describe().to_dict()
        },
        'time_coverage': {
            'time_span_days': (df_enhanced['Time'].max() - df_enhanced['Time'].min()) / (24 * 60 * 60),
            'transactions_per_hour': df_enhanced.groupby(df_enhanced['hour_of_day']).size().describe().to_dict()
        },
        'categorical_balance': {
            'transaction_types': df_enhanced['transaction_type'].value_counts().to_dict(),
            'card_types': df_enhanced['card_type'].value_counts().to_dict(),
            'regions': df_enhanced['region'].value_counts().to_dict(),
            'channels': df_enhanced['channel'].value_counts().to_dict()
        },
        'risk_score_separation': {
            'fraud_by_merchant': df_enhanced.groupby('merchant_id')['Class'].mean().describe().to_dict(),
            'fraud_by_customer': df_enhanced.groupby('customer_id')['Class'].mean().describe().to_dict()
        }
    }
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # Customer activity distribution
    plt.subplot(3, 3, 1)
    customer_activity = df_enhanced.groupby('customer_id').size()
    plt.hist(customer_activity, bins=50, alpha=0.7)
    plt.title('Customer Activity Distribution')
    plt.xlabel('Transactions per Customer')
    plt.ylabel('Frequency')
    
    # Merchant activity distribution
    plt.subplot(3, 3, 2)
    merchant_activity = df_enhanced.groupby('merchant_id').size()
    plt.hist(merchant_activity, bins=50, alpha=0.7)
    plt.title('Merchant Activity Distribution')
    plt.xlabel('Transactions per Merchant')
    plt.ylabel('Frequency')
    
    # Transactions by hour
    plt.subplot(3, 3, 3)
    hourly_transactions = df_enhanced.groupby('hour_of_day').size()
    plt.bar(hourly_transactions.index, hourly_transactions.values)
    plt.title('Transactions by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Transaction Count')
    
    # Transactions by day
    plt.subplot(3, 3, 4)
    daily_transactions = df_enhanced.groupby('day_of_week').size()
    plt.bar(daily_transactions.index, daily_transactions.values)
    plt.title('Transactions by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Transaction Count')
    
    # Transaction type distribution
    plt.subplot(3, 3, 5)
    df_enhanced['transaction_type'].value_counts().plot(kind='bar')
    plt.title('Transaction Type Distribution')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    
    # Card type distribution
    plt.subplot(3, 3, 6)
    df_enhanced['card_type'].value_counts().plot(kind='bar')
    plt.title('Card Type Distribution')
    plt.xlabel('Card Type')
    plt.ylabel('Count')
    
    # Region distribution
    plt.subplot(3, 3, 7)
    df_enhanced['region'].value_counts().plot(kind='bar')
    plt.title('Region Distribution')
    plt.xlabel('Region')
    plt.ylabel('Count')
    
    # Channel distribution
    plt.subplot(3, 3, 8)
    df_enhanced['channel'].value_counts().plot(kind='bar')
    plt.title('Channel Distribution')
    plt.xlabel('Channel')
    plt.ylabel('Count')
    
    # Risk score by class
    plt.subplot(3, 3, 9)
    df_enhanced.boxplot(column='merchant_fraud_rate', by='Class')
    plt.title('Merchant Fraud Rate by Class')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('risk_score_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Real-world characteristics analysis completed")
    return characteristics

def preprocess_data(filepath: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath: Path to the CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    preprocessor = DataPreprocessor()
    
    # Load and validate data
    df = preprocessor.load_data(filepath)
    preprocessor.validate_data(df)
    
    # Generate enterprise features
    df = preprocessor.generate_enterprise_features(df)
    
    # Engineer features
    df = preprocessor.engineer_features(df)
    
    # Encode categorical features
    df = preprocessor.encode_categorical_features(df, fit=True)
    
    # Scale features
    df = preprocessor.scale_features(df, fit=True)
    
    # Prepare for training
    X, y = preprocessor.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save preprocessor
    os.makedirs('artifacts', exist_ok=True)
    preprocessor.save_preprocessor('artifacts/preprocessor.joblib')
    
    logger.info(f"Preprocessing completed. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_file = "data/creditcard.csv"
    
    # Analyze dataset
    analysis = analyze_dataset(data_file)
    print("Dataset Analysis:", analysis)
    
    # Check real-world characteristics
    characteristics = check_real_world_characteristics(data_file)
    print("Real-world Characteristics:", characteristics)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data_file)
    print(f"Preprocessing completed: Train {X_train.shape}, Test {X_test.shape}") 