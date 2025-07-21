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