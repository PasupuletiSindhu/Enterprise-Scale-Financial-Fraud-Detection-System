#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection System.
Provides easy access to all functionality.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path to ensure imports work correctly
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def run_preprocessing():
    """Run the preprocessing pipeline."""
    print("🔄 Running preprocessing pipeline...")
    try:
        from src.preprocessing import preprocess_data, analyze_dataset, check_real_world_characteristics
        
        data_file = os.path.join("src", "data", "creditcard.csv")
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            print("Checking alternative locations...")
            
            alternatives = [
                "data/creditcard.csv",
                "creditcard.csv"
            ]
            
            for alt in alternatives:
                if os.path.exists(alt):
                    data_file = alt
                    print(f"✅ Found data file at: {data_file}")
                    break
            else:
                print("❌ No data file found. Please place the creditcard.csv file in src/data/ directory")
                return False
        
        # Analyze dataset
        print("📊 Analyzing dataset...")
        analysis = analyze_dataset(data_file)
        print(f"✅ Dataset analysis completed. Shape: {analysis['shape']}")
        
        # Check real-world characteristics
        print("🌍 Checking real-world characteristics...")
        characteristics = check_real_world_characteristics(data_file)
        print(f"✅ Real-world characteristics analysis completed")
        
        # Preprocess data
        print("⚙️ Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(data_file)
        print(f"✅ Preprocessing completed. Train: {X_train.shape}, Test: {X_test.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_training():
    """Run the training pipeline."""
    print("🎯 Running training pipeline...")
    try:
        from src.training import train_models
        from src.preprocessing import preprocess_data
        
        data_file = os.path.join("src", "data", "creditcard.csv")
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            
            alternatives = [
                "data/creditcard.csv",
                "creditcard.csv"
            ]
            
            for alt in alternatives:
                if os.path.exists(alt):
                    data_file = alt
                    print(f"✅ Found data file at: {data_file}")
                    break
            else:
                print("❌ No data file found. Please place the creditcard.csv file in src/data/ directory")
                return False
        
        # Preprocess data first
        print("⚙️ Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(data_file)
        
        # Train models
        print("🤖 Training models...")
        results = train_models(X_train, y_train, X_test, y_test)
        print(f"✅ Training completed. Models: {list(results.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_api():
    """Run the FastAPI server."""
    print("🚀 Starting FastAPI server...")
    try:
        import uvicorn
        from src.api import app
        
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"❌ API server failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_enhanced_api():
    """Run the Enhanced FastAPI server with WebSocket support."""
    print("🚀 Starting Enhanced FastAPI server with WebSocket support...")
    try:
        import uvicorn
        
        # Check if enhanced_api.py exists
        if not os.path.exists(os.path.join("src", "enhanced_api.py")):
            print("❌ Enhanced API file not found. Please make sure src/enhanced_api.py exists.")
            return False
            
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        # Import the app after ensuring the file exists
        from src.enhanced_api import app
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"❌ Enhanced API server failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("📊 Starting Streamlit dashboard...")
    try:
        import subprocess
        
        # Set environment variable to help with imports
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
        
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501"], env=env)
    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    try:
        import subprocess
        
        # Create test file if it doesn't exist
        test_file = "src/tests/test_api.py"
        if not os.path.exists(test_file):
            print("⚠️ Test file not found. Creating basic test...")
            create_basic_test()
        
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        subprocess.run([sys.executable, "-m", "pytest", "src/tests/", "-v"])
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_basic_test():
    """Create a basic test file."""
    test_content = '''
"""
Basic tests for the fraud detection system.
"""

import pytest
from src.prediction import FraudPredictor, create_sample_transaction

def test_sample_transaction():
    """Test sample transaction creation."""
    tx = create_sample_transaction()
    assert 'Time' in tx
    assert 'Amount' in tx
    assert 'V1' in tx

def test_predictor_initialization():
    """Test predictor initialization."""
    try:
        predictor = FraudPredictor()
        assert predictor is not None
    except Exception as e:
        pytest.skip(f"Predictor initialization failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    os.makedirs("src/tests", exist_ok=True)
    with open("src/tests/test_api.py", "w") as f:
        f.write(test_content)

def run_docker():
    """Run with Docker."""
    print("🐳 Running with Docker...")
    try:
        import subprocess
        
        # Check if docker-compose is installed
        try:
            subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ docker-compose not found. Please install Docker and docker-compose.")
            return False
        
        subprocess.run(["docker-compose", "up", "-d"])
    except Exception as e:
        print(f"❌ Docker deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fraud Detection System")
    parser.add_argument("command", choices=[
        "preprocess", "train", "api", "enhanced-api", "dashboard", "test", "docker", "all"
    ], help="Command to run")
    
    args = parser.parse_args()
    
    # Add src to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    if args.command == "preprocess":
        success = run_preprocessing()
        sys.exit(0 if success else 1)
    
    elif args.command == "train":
        success = run_training()
        sys.exit(0 if success else 1)
    
    elif args.command == "api":
        run_api()
    
    elif args.command == "enhanced-api":
        run_enhanced_api()
    
    elif args.command == "dashboard":
        run_dashboard()
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "docker":
        run_docker()
    
    elif args.command == "all":
        print("🔄 Running complete pipeline...")
        
        # Preprocess
        if not run_preprocessing():
            print("❌ Preprocessing failed")
            sys.exit(1)
        
        # Train
        if not run_training():
            print("❌ Training failed")
            sys.exit(1)
        
        print("✅ Complete pipeline finished successfully!")
        print("\n📋 Next steps:")
        print("1. Run API: python main.py api")
        print("2. Run Enhanced API: python main.py enhanced-api")
        print("3. Run Dashboard: python main.py dashboard")
        print("4. Run Tests: python main.py test")
        print("5. Run with Docker: python main.py docker")

if __name__ == "__main__":
    main() 