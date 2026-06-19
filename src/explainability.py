import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server/command-line execution
import matplotlib.pyplot as plt
import shap
import joblib

class ChurnExplainer:
    """
    Handles explainability for the XGBoost Churn prediction model using SHAP values.
    """
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self.preprocessor.feature_names
        
        # Handle cases where expected_value is a list or array (for binary classifiers)
        self.base_value = self.explainer.expected_value
        if isinstance(self.base_value, (list, np.ndarray)):
            if len(self.base_value) > 1:
                self.base_value = self.base_value[1] # standard for binary classification probability (log-odds)
            else:
                self.base_value = self.base_value[0]
        self.base_value = float(self.base_value)

    def explain_customer(self, X_processed_row):
        """
        Calculates feature attributions (SHAP values) for a single customer record.
        
        Returns:
        - shap_values: Numpy array of attributions
        - top_drivers: List of dicts with {'feature': name, 'impact': value, 'direction': 'Increase Churn' | 'Decrease Churn'}
        """
        # Ensure row is 2D DataFrame
        if isinstance(X_processed_row, pd.Series):
            X_processed_row = X_processed_row.to_frame().T
            
        shap_vals = self.explainer.shap_values(X_processed_row)
        
        # Extract the single row array
        if isinstance(shap_vals, list):
            # Binary classifier can return list of negative/positive class shap values
            row_shap = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            row_shap = shap_vals[0]
            
        # Match features with values
        features_dict = []
        for feat, val in zip(self.feature_names, row_shap):
            direction = "Increase Churn" if val > 0 else "Decrease Churn"
            features_dict.append({
                "feature": feat,
                "impact": float(val),
                "magnitude": abs(float(val)),
                "direction": direction
            })
            
        # Sort by impact magnitude descending
        features_dict = sorted(features_dict, key=lambda x: x["magnitude"], reverse=True)
        
        return row_shap, features_dict[:5]

    def save_waterfall_plot(self, X_processed_row, filepath):
        """
        Generates and saves a SHAP waterfall plot for a single customer.
        """
        if isinstance(X_processed_row, pd.Series):
            X_processed_row = X_processed_row.to_frame().T
            
        shap_vals = self.explainer.shap_values(X_processed_row)
        if isinstance(shap_vals, list):
            row_shap = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            row_shap = shap_vals[0]
            
        # Construct a SHAP Explanation object
        exp = shap.Explanation(
            values=row_shap,
            base_values=self.base_value,
            data=X_processed_row.values[0],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=10)
        plt.title("Customer Churn Risk Drivers (SHAP Attribution)", fontsize=14, pad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def save_summary_plot(self, X_processed_data, filepath):
        """
        Generates and saves a SHAP summary (beeswarm) plot for a dataset.
        """
        # Limit evaluation to at most 1000 samples to keep plot creation fast
        eval_data = X_processed_data.head(1000)
        shap_vals = self.explainer.shap_values(eval_data)
        
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_vals = shap_vals[1]
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, eval_data, feature_names=self.feature_names, show=False, max_display=12)
        plt.title("Global Churn Feature Importances (SHAP Summary)", fontsize=14, pad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
