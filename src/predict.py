import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib

# Local imports
from preprocess import ChurnPreprocessor
from segmentation import CustomerSegmenter
from clv import calculate_clv
from revenue_risk import calculate_customer_revenue_risk
from explainability import ChurnExplainer
from recommendation_engine import RetentionRecommendationEngine

class CustomerInferencePipeline:
    """
    Unified pipeline to perform end-to-end predictions, segmentation, business risk valuation, 
    explainability, and retention strategy generation for customer profiles.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Load preprocessor, segmenter, and churn classifier
        self.preprocessor = ChurnPreprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.segmenter = CustomerSegmenter.load(os.path.join(models_dir, "segmenter.joblib"))
        self.churn_model = joblib.load(os.path.join(models_dir, "churn_model.joblib"))
        self.explainer = ChurnExplainer(
            model_path=os.path.join(models_dir, "churn_model.joblib"),
            preprocessor_path=os.path.join(models_dir, "preprocessor.joblib")
        )
        self.recommender = RetentionRecommendationEngine()

    def analyze_customer(self, raw_customer_dict):
        """
        Analyzes a single customer record and returns comprehensive predictive and descriptive metrics.
        """
        # Convert single dictionary to DataFrame matching the preprocess input format
        df_raw = pd.DataFrame([raw_customer_dict])
        
        # 1. Preprocess and engineer features
        X_processed = self.preprocessor.transform(df_raw)
        
        # 2. Churn prediction (probability)
        churn_prob = float(self.churn_model.predict_proba(X_processed)[0, 1])
        
        # 3. Customer Segment assignment
        segment = self.segmenter.get_business_segments(X_processed)[0]
        
        # 4. CLV estimation
        clv_series = calculate_clv(df_raw, churn_probabilities=[churn_prob])
        clv = float(clv_series.iloc[0])
        
        # 5. Revenue at Risk calculation
        revenue_at_risk = float(calculate_customer_revenue_risk(clv, churn_prob))
        
        # 6. SHAP Explainability (extract drivers)
        _, top_drivers = self.explainer.explain_customer(X_processed.iloc[0])
        
        # Map feature names to human-readable drivers for executive report
        driver_mapping = {
            "Complaints": "Multiple Complaints",
            "ComplaintsPerTenure": "High Complaint Rate",
            "EngagementScore": "Low Engagement",
            "IsActiveMember": "Inactive Member",
            "Balance": "Reduced Balance",
            "BalanceToSalaryRatio": "Low Balance Relative to Salary",
            "TransactionVolume": "Low Transaction Frequency",
            "TransactionAmount": "Low Monthly Spend",
            "NumOfProducts": "Product Count Mismatch",
            "Age": "Demographic Risk Factor"
        }
        
        display_drivers = []
        for drv in top_drivers:
            # Only count features contributing to higher churn risk
            if drv["direction"] == "Increase Churn":
                clean_name = driver_mapping.get(drv["feature"], drv["feature"].replace("_", " "))
                display_drivers.append(clean_name)
                
        # Fallback if no positive drivers (low risk)
        if not display_drivers:
            display_drivers = ["Stable Relationship Indicators"]
            
        # 7. Recommendations
        recs = self.recommender.generate_recommendations(churn_prob, segment, top_drivers)
        
        return {
            "CustomerID": raw_customer_dict.get("CustomerID", "UNKNOWN"),
            "ChurnRiskScore": churn_prob,
            "Segment": segment,
            "CLV": clv,
            "RevenueAtRisk": revenue_at_risk,
            "TopDrivers": display_drivers[:3],
            "Recommendations": recs
        }

    def print_executive_report(self, analysis_result):
        """
        Prints a professionally formatted business report for a single customer.
        """
        import sys
        # Try to reconfigure stdout for UTF-8 to display the Rupee symbol correctly
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            currency_symbol = "₹"
        except Exception:
            currency_symbol = "INR "

        try:
            print("\n" + "=" * 45)
            print("     EXECUTIVE RETENTION REPORT")
            print("=" * 45)
            print(f"Customer ID:       {analysis_result['CustomerID']}")
            print(f"Customer Risk Score: {analysis_result['ChurnRiskScore']:.0%}")
            print(f"\nCustomer Segment:\n{analysis_result['Segment']}")
            print(f"\nCustomer Lifetime Value:\n{currency_symbol}{analysis_result['CLV']:,.0f}")
            print(f"\nRevenue At Risk:\n{currency_symbol}{analysis_result['RevenueAtRisk']:,.0f}")
            
            print("\nTop Drivers:")
            for driver in analysis_result["TopDrivers"]:
                print(f"* {driver}")
                
            print("\nRecommended Actions:")
            for rec in analysis_result["Recommendations"]:
                print(f"* [{rec['Priority']}] {rec['Action']}")
                print(f"  --> {rec['Description']}")
            print("=" * 45 + "\n")
        except UnicodeEncodeError:
            # Full fallback with ascii-safe currency string
            currency_symbol = "INR "
            print("\n" + "=" * 45)
            print("     EXECUTIVE RETENTION REPORT")
            print("=" * 45)
            print(f"Customer ID:       {analysis_result['CustomerID']}")
            print(f"Customer Risk Score: {analysis_result['ChurnRiskScore']:.0%}")
            print(f"\nCustomer Segment:\n{analysis_result['Segment']}")
            print(f"\nCustomer Lifetime Value:\n{currency_symbol}{analysis_result['CLV']:,.0f}")
            print(f"\nRevenue At Risk:\n{currency_symbol}{analysis_result['RevenueAtRisk']:,.0f}")
            
            print("\nTop Drivers:")
            for driver in analysis_result["TopDrivers"]:
                print(f"* {driver}")
                
            print("\nRecommended Actions:")
            for rec in analysis_result["Recommendations"]:
                print(f"* [{rec['Priority']}] {rec['Action']}")
                print(f"  --> {rec['Description']}")
            print("=" * 45 + "\n")

if __name__ == "__main__":
    # Test inference pipeline
    pipeline = CustomerInferencePipeline()
    
    # 1. Simulate a high risk customer profile
    high_risk_cust = {
        "CustomerID": "CUST-99001",
        "Age": 48,
        "Gender": "Female",
        "Geography": "Mumbai",
        "TenureMonths": 12,
        "Balance": 1800000.0, # High balance
        "NumOfProducts": 3,   # 3 products (trigger confusion/risk)
        "HasCrCard": 1,
        "IsActiveMember": 0,  # Inactive
        "EstimatedSalary": 1200000.0,
        "TransactionVolume": 2,  # Low transaction volume
        "TransactionAmount": 15000.0, # Low transaction amount
        "Complaints": 4       # Multiple complaints!
    }
    
    print("Analyzing High-Risk Customer...")
    res_high = pipeline.analyze_customer(high_risk_cust)
    pipeline.print_executive_report(res_high)
    
    # 2. Simulate a low risk loyal customer profile
    low_risk_cust = {
        "CustomerID": "CUST-99002",
        "Age": 35,
        "Gender": "Male",
        "Geography": "Bengaluru",
        "TenureMonths": 72,
        "Balance": 2200000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,  # Active
        "EstimatedSalary": 2400000.0,
        "TransactionVolume": 28,
        "TransactionAmount": 120000.0,
        "Complaints": 0       # No complaints
    }
    
    print("Analyzing Low-Risk Customer...")
    res_low = pipeline.analyze_customer(low_risk_cust)
    pipeline.print_executive_report(res_low)
