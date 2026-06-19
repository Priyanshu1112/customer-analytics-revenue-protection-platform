import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# Local imports
from preprocess import ChurnPreprocessor
from clv import calculate_clv
from revenue_risk import calculate_customer_revenue_risk, calculate_portfolio_revenue_risk

class ExecutiveDashboard:
    """
    Compiles portfolio-wide risk aggregates and generates executive-ready visual reports.
    """
    def __init__(self, models_dir="models"):
        self.preprocessor = ChurnPreprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.churn_model = joblib.load(os.path.join(models_dir, "churn_model.joblib"))
        self.segmenter = joblib.load(os.path.join(models_dir, "segmenter.joblib"))

    def compile_metrics(self, df_raw):
        """
        Runs the full predictive pipeline on a DataFrame of customers and compiles aggregates.
        """
        df = df_raw.copy()
        
        # 1. Transform features
        X_raw = df.drop(columns=["CustomerID", "Churn"], errors="ignore")
        X_processed = self.preprocessor.transform(X_raw)
        
        # 2. Get churn probabilities and segments
        df["ChurnProb"] = self.churn_model.predict_proba(X_processed)[:, 1]
        df["Segment"] = self.segmenter.get_business_segments(X_processed)
        
        # 3. Calculate financial metrics
        df["CLV"] = calculate_clv(df, df["ChurnProb"])
        df["RevenueAtRisk"] = calculate_customer_revenue_risk(df["CLV"], df["ChurnProb"])
        
        # 4. Aggregate metrics
        portfolio_summary = calculate_portfolio_revenue_risk(df)
        
        return df, portfolio_summary

    def generate_text_dashboard(self, portfolio_summary):
        """
        Prints a text-based ASCII executive dashboard.
        """
        # Set stdout to UTF-8 or use safe chars
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            currency = "₹"
        except Exception:
            currency = "INR "
            
        print("\n" + "=" * 65)
        print("          EXECUTIVE CUSTOMER ANALYTICS & RISK DASHBOARD")
        print("=" * 65)
        print(f"Total Portfolio Customers:  {sum([s['Customer_Count'] for s in portfolio_summary['Segment_Risk'].values()]):,}")
        print(f"Total Portfolio Valuation (CLV): {currency}{portfolio_summary['Total_Portfolio_CLV']:,.2f}")
        print(f"Total Revenue At Risk:           {currency}{portfolio_summary['Total_Revenue_At_Risk']:,.2f}")
        print(f"Portfolio Risk Exposure Rate:    {portfolio_summary['Portfolio_Risk_Percentage']}%")
        print("-" * 65)
        
        print(f"{'Segment Name':<25} | {'Size':<6} | {'Avg Risk':<8} | {'Rev at Risk':<15}")
        print("-" * 65)
        for seg, data in portfolio_summary["Segment_Risk"].items():
            avg_risk_str = f"{data['Total_Revenue_At_Risk'] / data['Total_CLV']:.1%}"
            print(f"{seg:<25} | {data['Customer_Count']:<6} | {avg_risk_str:<8} | {currency}{data['Total_Revenue_At_Risk']:<14,.0f}")
        print("=" * 65 + "\n")

    def save_visual_dashboard(self, df_scored, filepath):
        """
        Generates and saves a publication-quality executive dashboard dashboard figure.
        """
        # Styling parameters for premium design
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.family'] = 'sans-serif'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle("Customer Analytics & Revenue Protection Executive Dashboard", fontsize=18, fontweight='bold', color='#1E293B', y=0.98)
        
        # Color palette
        colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]
        
        # 1. Churn Risk Distribution (Subplot top-left)
        ax = axes[0, 0]
        ax.hist(df_scored["ChurnProb"] * 100, bins=25, color="#3B82F6", edgecolor="white", alpha=0.85)
        ax.axvline(30, color="#EF4444", linestyle="--", linewidth=1.5, label="High Risk Threshold (30%)")
        ax.set_title("Customer Distribution by Churn Risk Score", fontsize=12, fontweight='semibold', color='#1E293B')
        ax.set_xlabel("Predicted Churn Probability (%)", fontsize=10)
        ax.set_ylabel("Customer Count", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # 2. Customers by Segment (Subplot top-right)
        ax = axes[0, 1]
        seg_counts = df_scored["Segment"].value_counts()
        ax.pie(
            seg_counts, 
            labels=seg_counts.index, 
            autopct='%1.1f%%', 
            colors=["#F59E0B", "#3B82F6", "#10B981", "#EF4444"], 
            startangle=140,
            textprops={'fontsize': 10, 'weight': 'bold', 'color': '#1E293B'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
        )
        ax.set_title("Portfolio Breakdown by Customer Segment", fontsize=12, fontweight='semibold', color='#1E293B')
        
        # 3. Revenue at Risk by Segment (Subplot bottom-left)
        ax = axes[1, 0]
        seg_risk = df_scored.groupby("Segment")["RevenueAtRisk"].sum() / 1e6 # convert to Millions
        seg_risk.plot(kind="bar", color=["#EF4444", "#3B82F6", "#10B981", "#F59E0B"], ax=ax, edgecolor="white", alpha=0.85)
        ax.set_title("Revenue at Risk by Segment (INR Millions)", fontsize=12, fontweight='semibold', color='#1E293B')
        ax.set_xlabel("")
        ax.set_ylabel("Revenue at Risk (₹ Millions)", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # 4. XGBoost Feature Importance (Subplot bottom-right)
        ax = axes[1, 1]
        feature_importance = self.churn_model.feature_importances_
        feature_names = self.preprocessor.feature_names
        
        # Create a DataFrame of feature importances
        df_feat = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values("Importance", ascending=True).tail(8) # Top 8 features
        
        # Format names for dashboard presentation
        df_feat["DisplayName"] = df_feat["Feature"].apply(
            lambda x: {
                "Complaints": "Complaints Count",
                "ComplaintsPerTenure": "Complaint Frequency",
                "IsActiveMember": "Membership Inactivity",
                "EngagementScore": "Engagement Score",
                "NumOfProducts": "Product Count",
                "Balance": "Account Balance",
                "BalanceToSalaryRatio": "Balance-to-Salary Ratio",
                "TransactionVolume": "Transaction Volume",
                "Age": "Customer Age",
                "TenureMonths": "Tenure (Months)"
            }.get(x, x.replace("_", " "))
        )
        
        ax.barh(df_feat["DisplayName"], df_feat["Importance"], color="#10B981", edgecolor="white", alpha=0.85)
        ax.set_title("Top Churn Drivers (XGBoost Feature Importance)", fontsize=12, fontweight='semibold', color='#1E293B')
        ax.set_xlabel("Relative Importance Weight", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Executive Visual Dashboard successfully saved to: {filepath}")

if __name__ == "__main__":
    # Test dashboard compilation on full dataset
    data_path = os.path.join("data", "customer_churn_data.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        dashboard = ExecutiveDashboard()
        df_scored, summary = dashboard.compile_metrics(df_raw)
        dashboard.generate_text_dashboard(summary)
        dashboard.save_visual_dashboard(df_scored, os.path.join("screenshots", "executive_dashboard.png"))
    else:
        print("Data path not found. Please run preprocess.py first to generate the dataset.")
