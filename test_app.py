#!/usr/bin/env python3
"""
Test script to verify the IoT Anomaly Detection application works correctly.
"""

import pandas as pd
import numpy as np
from app import validate_columns, engineer_features, fit_and_predict
import extra_features as ef

def test_application():
    """Test the main application functions"""
    print("🧪 Testing IoT Anomaly Detection Application...")
    
    # Test 1: Load sample data
    try:
        df = pd.read_csv("iot_anomaly_dataset.csv")
        print("✅ Sample data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load sample data: {e}")
        return False
    
    # Test 2: Validate columns
    is_valid, missing = validate_columns(df)
    if is_valid:
        print("✅ Column validation passed")
    else:
        print(f"❌ Column validation failed. Missing: {missing}")
        return False
    
    # Test 3: Feature engineering
    try:
        features, labels = engineer_features(df)
        print(f"✅ Feature engineering successful. Features shape: {features.shape}")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    # Test 4: Model training and prediction
    try:
        pred, X_scaled, model, scaler = fit_and_predict(features, labels, contamination=0.05)
        print(f"✅ Model training successful. Predictions shape: {pred.shape}")
        print(f"   Anomalies detected: {pred.sum()}")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False
    
    # Test 5: Extra features
    try:
        # Test anomaly logging
        ef.log_anomaly_to_csv({"test": "data"}, "TestAnomaly")
        print("✅ Anomaly logging works")
        
        # Test email alert (should skip if no credentials)
        ef.send_email_alert("Test", "Test message", "test@example.com")
        print("✅ Email alert function works")
        
    except Exception as e:
        print(f"❌ Extra features failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The application is working correctly.")
    print("\nTo run the application:")
    print("1. Streamlit app: streamlit run app.py")
    print("2. Flask dashboard: python dashboard.py")
    
    return True

if __name__ == "__main__":
    test_application()
