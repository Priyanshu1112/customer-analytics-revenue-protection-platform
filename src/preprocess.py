import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generates a synthetic banking customer dataset with realistic correlations for churn prediction.
    """
    np.random.seed(random_state)
    
    # Customer IDs
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # Demographics
    ages = np.random.normal(loc=41, scale=12, size=n_samples).astype(int)
    ages = np.clip(ages, 18, 85)
    
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.54, 0.46])
    geographies = np.random.choice(["Mumbai", "Delhi", "Bengaluru"], size=n_samples, p=[0.45, 0.35, 0.20])
    
    # Bank relationship details
    tenures = np.random.randint(0, 120, size=n_samples) # in months
    
    # Financial details (gamma distributed to be right-skewed)
    balances = np.random.gamma(shape=3.0, scale=80000, size=n_samples)
    # Give some customers zero balances (realistic for churners)
    zero_balance_mask = np.random.choice([True, False], size=n_samples, p=[0.15, 0.85])
    balances[zero_balance_mask] = 0.0
    
    num_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.50, 0.40, 0.08, 0.02])
    has_cr_card = np.random.choice([0, 1], size=n_samples, p=[0.30, 0.70])
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52])
    
    estimated_salaries = np.random.uniform(300000, 2500000, size=n_samples)
    
    # Transactions in the last month
    transaction_volumes = np.random.poisson(lam=12, size=n_samples)
    # Correlate transaction amount with salary and activity status
    base_trans_amt = estimated_salaries * 0.03
    transaction_amounts = np.random.normal(loc=base_trans_amt, scale=base_trans_amt * 0.3)
    transaction_amounts = np.clip(transaction_amounts, 0, None)
    # Inactive members spend less
    inactive_mask = (is_active_member == 0)
    transaction_amounts[inactive_mask] *= np.random.uniform(0.2, 0.6, size=sum(inactive_mask))
    transaction_volumes[inactive_mask] = (transaction_volumes[inactive_mask] * 0.4).astype(int)
    
    # Customer complaints in last 6 months
    complaints = np.random.poisson(lam=0.3, size=n_samples)
    # Limit complaints to maximum of 5
    complaints = np.clip(complaints, 0, 5)
    
    # Churn definition logic based on logical business rules (probability model)
    # Base churn risk logit
    logit = (
        -1.8
        + 0.04 * (ages - 40)               # Older customers churn slightly more in this market
        - 0.015 * (tenures / 12.0)          # Longer tenure reduces churn
        - 0.0000015 * balances              # Lower balance increases churn
        - 1.1 * is_active_member            # Active members are much less likely to churn
        + 1.5 * complaints                  # Complaints strongly drive churn
        + 0.4 * (num_products == 1).astype(int) # Single product customers are less sticky
        + 1.2 * (num_products >= 3).astype(int) # Multi-product (>2) can signify confusion/unmet needs here
        - 0.3 * (estimated_salaries / 1000000.0) # High salary reduces churn slightly
    )
    
    # Convert logit to probability
    churn_probs = 1 / (1 + np.exp(-logit))
    
    # Sample actual Churn binary label
    churn = np.random.binomial(1, churn_probs)
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Geography": geographies,
        "TenureMonths": tenures,
        "Balance": balances,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salaries,
        "TransactionVolume": transaction_volumes,
        "TransactionAmount": transaction_amounts,
        "Complaints": complaints,
        "Churn": churn
    })
    
    return df

class ChurnPreprocessor:
    """
    Handles cleaning, encoding, and feature engineering for the Customer Analytics platform.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.num_cols = [
            "Age", "TenureMonths", "Balance", "NumOfProducts", 
            "EstimatedSalary", "TransactionVolume", "TransactionAmount", 
            "Complaints", "BalanceToSalaryRatio", "AverageTransactionSize", 
            "ComplaintsPerTenure", "EngagementScore", "BalancePerProduct"
        ]
        self.cat_cols = ["Gender", "Geography"]
        self.bin_cols = ["HasCrCard", "IsActiveMember"]
        self.feature_names = None

    def _engineer_features(self, df):
        """
        Creates new business-relevant features.
        """
        df_copy = df.copy()
        
        # 1. Balance-to-Salary Ratio
        df_copy["BalanceToSalaryRatio"] = df_copy["Balance"] / (df_copy["EstimatedSalary"] + 1e-5)
        
        # 2. Average Transaction Size
        df_copy["AverageTransactionSize"] = df_copy["TransactionAmount"] / (df_copy["TransactionVolume"] + 1e-5)
        
        # 3. Complaints Per Tenure Month
        df_copy["ComplaintsPerTenure"] = df_copy["Complaints"] / (df_copy["TenureMonths"] + 1.0)
        
        # 4. Engagement Score (0 to 6 index of activity)
        df_copy["EngagementScore"] = (
            df_copy["IsActiveMember"] * 2 
            + df_copy["HasCrCard"] * 1 
            + (df_copy["TransactionVolume"] > 10).astype(int) * 2 
            + (df_copy["NumOfProducts"] == 2).astype(int) * 1
        )
        
        # 5. Balance Per Product
        df_copy["BalancePerProduct"] = df_copy["Balance"] / df_copy["NumOfProducts"]
        
        return df_copy

    def fit(self, X, y=None):
        """
        Fits encoder and scaler on feature data.
        """
        # Engineer features first
        X_engineered = self._engineer_features(X)
        
        # Fit Categorical Encoder
        self.encoder.fit(X_engineered[self.cat_cols])
        
        # Transform categories to get column names
        encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + self.bin_cols + encoded_cat_names
        
        # Fit Scaler on Numerical Features
        self.scaler.fit(X_engineered[self.num_cols])
        return self

    def transform(self, X):
        """
        Transforms raw feature data.
        """
        X_engineered = self._engineer_features(X)
        
        # Scale numerical columns
        scaled_num = self.scaler.transform(X_engineered[self.num_cols])
        df_scaled_num = pd.DataFrame(scaled_num, columns=self.num_cols, index=X.index)
        
        # Encode categorical columns
        encoded_cat = self.encoder.transform(X_engineered[self.cat_cols])
        encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols)
        df_encoded_cat = pd.DataFrame(encoded_cat, columns=encoded_cat_names, index=X.index)
        
        # Combine everything
        df_bin = X_engineered[self.bin_cols]
        X_processed = pd.concat([df_scaled_num, df_bin, df_encoded_cat], axis=1)
        
        # Return with consistent column order
        return X_processed[self.feature_names]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def save(self, filepath):
        """Saves the preprocessor object using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Loads a preprocessor object using joblib."""
        return joblib.load(filepath)

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Generating synthetic customer dataset (5,000 samples)...")
    raw_data = generate_synthetic_data(n_samples=5000)
    raw_data_path = os.path.join("data", "customer_churn_data.csv")
    raw_data.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to: {raw_data_path}")
    print(f"Churn rate in generated data: {raw_data['Churn'].mean():.2%}")
    
    print("\nFitting and saving ChurnPreprocessor...")
    preprocessor = ChurnPreprocessor()
    X = raw_data.drop(columns=["CustomerID", "Churn"])
    X_processed = preprocessor.fit_transform(X)
    
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor fitted and saved to: {preprocessor_path}")
    print(f"Processed feature matrix shape: {X_processed.shape}")
    print("Pre-processed features list:")
    print(preprocessor.feature_names)

