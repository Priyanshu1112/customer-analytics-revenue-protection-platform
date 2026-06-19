import numpy as np
import pandas as pd

def calculate_customer_revenue_risk(clv, churn_probability):
    """
    Calculates the Revenue at Risk for an individual customer.
    
    Revenue at Risk = Churn Probability * Customer Lifetime Value (CLV)
    """
    return np.round(churn_probability * clv, 2)

def calculate_portfolio_revenue_risk(df_with_clv_and_risk):
    """
    Calculates portfolio-wide and segment-wide Revenue at Risk metrics.
    
    Returns a dictionary of aggregated metrics:
    - Total CLV of Portfolio
    - Total Revenue at Risk
    - Portfolio-level Risk Percentage
    - Revenue at Risk by Segment
    """
    total_clv = df_with_clv_and_risk["CLV"].sum()
    total_revenue_at_risk = df_with_clv_and_risk["RevenueAtRisk"].sum()
    risk_percentage = (total_revenue_at_risk / total_clv) if total_clv > 0 else 0.0
    
    # Segment-level aggregates if Segment is present in the DataFrame
    segment_risk = {}
    if "Segment" in df_with_clv_and_risk.columns:
        segment_summary = df_with_clv_and_risk.groupby("Segment").agg(
            Total_CLV=("CLV", "sum"),
            Total_Revenue_At_Risk=("RevenueAtRisk", "sum"),
            Customer_Count=("CustomerID", "count")
        )
        segment_summary["Risk_Percentage"] = segment_summary["Total_Revenue_At_Risk"] / segment_summary["Total_CLV"]
        segment_risk = segment_summary.to_dict(orient="index")
        
    return {
        "Total_Portfolio_CLV": np.round(total_clv, 2),
        "Total_Revenue_At_Risk": np.round(total_revenue_at_risk, 2),
        "Portfolio_Risk_Percentage": np.round(risk_percentage * 100, 2),
        "Segment_Risk": segment_risk
    }
