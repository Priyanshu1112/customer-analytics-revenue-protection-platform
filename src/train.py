import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# Local imports
from preprocess import ChurnPreprocessor, generate_synthetic_data
from segmentation import CustomerSegmenter

def run_training_pipeline():
    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    data_path = os.path.join("data", "customer_churn_data.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Generating a new dataset...")
        raw_df = generate_synthetic_data(n_samples=5000)
        os.makedirs("data", exist_ok=True)
        raw_df.to_csv(data_path, index=False)
    else:
        raw_df = pd.read_csv(data_path)
    
    print(f"Loaded dataset: {raw_df.shape[0]} customers, {raw_df.shape[1]} features.")
    
    # 2. Initialize and fit preprocessor (or load if already fitted)
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        print("Loading existing preprocessor...")
        preprocessor = ChurnPreprocessor.load(preprocessor_path)
    else:
        print("Fitting a new preprocessor...")
        preprocessor = ChurnPreprocessor()
        X_raw = raw_df.drop(columns=["CustomerID", "Churn"])
        preprocessor.fit(X_raw)
        preprocessor.save(preprocessor_path)
        
    # 3. Transform features
    X_raw = raw_df.drop(columns=["CustomerID", "Churn"])
    y = raw_df["Churn"]
    X_processed = preprocessor.transform(X_raw)
    
    # 4. Fit Customer Segmentation (K-Means)
    print("\nTraining K-Means Customer Segmentation model...")
    segmenter_path = os.path.join("models", "segmenter.joblib")
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(X_processed, raw_df)
    segmenter.save(segmenter_path)
    
    # Predict segments
    segments = segmenter.get_business_segments(X_processed)
    raw_df["Segment"] = segments
    print("Customer segments generated and mapped successfully.")
    print(raw_df["Segment"].value_counts())
    
    # 5. Train Churn Prediction Model (XGBoost)
    print("\nSplitting dataset into train and validation sets (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    print(f"Training churn rate: {y_train.mean():.2%}, Validation churn rate: {y_val.mean():.2%}")
    
    print("\nTraining XGBoost Churn Classifier...")
    # Scale positive weights if class imbalance is high (optional, but good practice)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    churn_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    
    churn_model.fit(X_train, y_train)
    
    # 6. Evaluate Churn Model
    y_pred = churn_model.predict(X_val)
    y_prob = churn_model.predict_proba(X_val)[:, 1]
    
    print("\n" + "=" * 40)
    print("Model Evaluation Metrics (Validation Set)")
    print("=" * 40)
    
    # Print metrics
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score (Average Precision): {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save churn model
    model_path = os.path.join("models", "churn_model.joblib")
    joblib.dump(churn_model, model_path)
    print(f"Churn XGBoost model successfully saved to: {model_path}")
    
    # 7. Print Segment Analysis
    print("\n" + "=" * 40)
    print("Segment Profiling Analysis")
    print("=" * 40)
    
    # Predict churn probability for the entire dataset to compute risk metrics by segment
    full_probs = churn_model.predict_proba(X_processed)[:, 1]
    raw_df["ChurnProb"] = full_probs
    
    segment_profile = raw_df.groupby("Segment").agg(
        Count=("CustomerID", "count"),
        MeanAge=("Age", "mean"),
        MeanBalance=("Balance", "mean"),
        MeanVolume=("TransactionVolume", "mean"),
        MeanComplaints=("Complaints", "mean"),
        ObservedChurnRate=("Churn", "mean"),
        PredictedChurnRisk=("ChurnProb", "mean")
    ).round(2)
    
    print(segment_profile)
    print("=" * 60)

if __name__ == "__main__":
    run_training_pipeline()
