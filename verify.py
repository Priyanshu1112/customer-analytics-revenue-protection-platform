import os
import pandas as pd
import numpy as np

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from src.preprocess import generate_synthetic_data, ChurnPreprocessor
from src.train import run_training_pipeline
from src.predict import CustomerInferencePipeline
from src.explainability import ChurnExplainer
from src.dashboard_metrics import ExecutiveDashboard

def main():
    print("=" * 60)
    print("RUNNING END-TO-END VERIFICATION AND ASSETS GENERATION")
    print("=" * 60)
    
    # Ensure directories
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Run model training pipeline (will also create data if missing)
    run_training_pipeline()
    
    # 2. Run Inference Pipeline on simulated high-risk customer
    print("\nRunning test inference...")
    pipeline = CustomerInferencePipeline()
    
    high_risk_cust = {
        "CustomerID": "CUST-99001",
        "Age": 48,
        "Gender": "Female",
        "Geography": "Mumbai",
        "TenureMonths": 12,
        "Balance": 1800000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 1200000.0,
        "TransactionVolume": 2,
        "TransactionAmount": 15000.0,
        "Complaints": 4
    }
    
    res = pipeline.analyze_customer(high_risk_cust)
    pipeline.print_executive_report(res)
    
    # 3. Generate SHAP explainability plots
    print("\nGenerating SHAP explanation plots...")
    explainer = ChurnExplainer(
        model_path="models/churn_model.joblib",
        preprocessor_path="models/preprocessor.joblib"
    )
    
    # Load dataset to generate global summary
    df_raw = pd.read_csv("data/customer_churn_data.csv")
    X_raw = df_raw.drop(columns=["CustomerID", "Churn"])
    X_processed = pipeline.preprocessor.transform(X_raw)
    
    # Save SHAP summary plot
    summary_path = os.path.join("screenshots", "shap_summary.png")
    explainer.save_summary_plot(X_processed, summary_path)
    print(f"SHAP summary plot saved to: {summary_path}")
    
    # Save SHAP waterfall plot for the high-risk customer
    waterfall_path = os.path.join("screenshots", "shap_waterfall_cust_99001.png")
    X_processed_cust = pipeline.preprocessor.transform(pd.DataFrame([high_risk_cust]))
    explainer.save_waterfall_plot(X_processed_cust.iloc[0], waterfall_path)
    print(f"SHAP waterfall plot saved to: {waterfall_path}")
    
    # 4. Generate Executive Dashboard Visual
    print("\nGenerating executive business metrics dashboard...")
    dashboard = ExecutiveDashboard()
    df_scored, summary = dashboard.compile_metrics(df_raw)
    dashboard.generate_text_dashboard(summary)
    
    dashboard_visual_path = os.path.join("screenshots", "executive_dashboard.png")
    dashboard.save_visual_dashboard(df_scored, dashboard_visual_path)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETED SUCCESSFULLY. ALL ASSETS GENERATED.")
    print("=" * 60)

if __name__ == "__main__":
    main()
