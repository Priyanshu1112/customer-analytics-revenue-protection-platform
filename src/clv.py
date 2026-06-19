import numpy as np
import pandas as pd

def estimate_annual_profit(balance, transaction_volume, transaction_amount, estimated_salary):
    """
    Estimates the annual profit margin a customer generates for the bank.
    
    Formula components:
    - Net Interest Income (NIM): 3% on Balance
    - Fee Income: ₹10 per transaction + 0.1% of total transaction amount
    - Product Cross-sell Margin: Estimated from salary (0.05% of salary)
    - Servicing Cost: ₹1,500 base cost per customer
    """
    nii = balance * 0.03
    fee_income = (transaction_volume * 10.0 + transaction_amount * 0.001) * 12.0
    cross_sell = estimated_salary * 0.0005
    servicing_cost = 1500.0
    
    annual_profit = nii + fee_income + cross_sell - servicing_cost
    
    # Floor the annual profit at a baseline of ₹500 to keep CLV calculations meaningful 
    # even for inactive/low-balance customers.
    return np.maximum(annual_profit, 500.0)

def calculate_clv(df, churn_probabilities=None, discount_rate=0.10):
    """
    Calculates Customer Lifetime Value (CLV) using the Churn-adjusted Discounting Model:
    
    CLV = Annual Profit / (Churn Probability + Discount Rate)
    
    Parameters:
    - df: DataFrame containing Balance, TransactionVolume, TransactionAmount, EstimatedSalary
    - churn_probabilities: Array-like of churn probabilities from the model. 
                           If None, uses a baseline churn risk.
    - discount_rate: Cost of capital / discount factor (default: 10%)
    """
    annual_profit = estimate_annual_profit(
        df["Balance"], 
        df["TransactionVolume"], 
        df["TransactionAmount"],
        df["EstimatedSalary"]
    )
    
    if churn_probabilities is None:
        # Default fallback: check if Churn column is present, otherwise use 15% flat rate
        if "Churn" in df.columns:
            churn_probs = np.where(df["Churn"] == 1, 0.85, 0.15)
        else:
            churn_probs = np.full(len(df), 0.15)
    else:
        churn_probs = np.array(churn_probabilities)
        
    # Standard CLV formula adjusted for customer churn risk
    clv = annual_profit / (churn_probs + discount_rate)
    
    return np.round(clv, 2)
