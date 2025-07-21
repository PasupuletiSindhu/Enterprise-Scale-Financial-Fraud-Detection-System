#!/usr/bin/env python3
"""
Simple test for real-time processor with existing models.
"""

import sys
import os
sys.path.append('src')

def test_model_files():
    """Test if model files exist."""
    print("🔍 Checking model files...")
    
    model_files = [
        'artifacts/xgboost.joblib',
        'artifacts/preprocessor.joblib',
        'artifacts/evaluation_results.joblib'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - EXISTS")
        else:
            print(f"❌ {file_path} - MISSING")
    
    return all(os.path.exists(f) for f in model_files)

def test_realtime_processor():
    """Test real-time processor initialization."""
    print("\n🚀 Testing real-time processor...")
    
    try:
        from realtime_processor import RealTimeFraudProcessor
        
        processor = RealTimeFraudProcessor()
        print("✅ Real-time processor initialized successfully!")
        
        # Test sample transaction
        test_transaction = processor.generate_sample_transaction()
        print(f"✅ Generated test transaction: {test_transaction['id']}")
        
        # Test prediction
        result = processor.predict_fraud(test_transaction)
        print(f"✅ Prediction result: {result['fraud_probability']:.3f} ({result['model_used']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time processor test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 REAL-TIME PROCESSOR TEST")
    print("=" * 50)
    
    # Test model files
    models_exist = test_model_files()
    
    if models_exist:
        # Test real-time processor
        rt_works = test_realtime_processor()
        
        if rt_works:
            print("\n🎉 All tests passed! Real-time processor is ready.")
        else:
            print("\n⚠️ Real-time processor test failed.")
    else:
        print("\n❌ Model files missing. Please run training first.") 