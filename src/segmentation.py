import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

class CustomerSegmenter:
    """
    Wraps KMeans clustering to segment customers based on their financial and transaction features.
    Provides business-relevant labels for each cluster.
    """
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_mapping = {} # Maps cluster index to business label
        
        # Features to use for clustering (financial & transactional)
        self.clustering_features = [
            "Balance", "EstimatedSalary", "TransactionVolume", 
            "TransactionAmount", "EngagementScore", "Complaints"
        ]

    def fit(self, X_processed, df_raw):
        """
        Fits KMeans on the preprocessed features and analyzes the raw customer data 
        to label clusters with business terms.
        """
        # Fit K-Means on the selected processed feature columns
        X_cluster_input = X_processed[self.clustering_features]
        self.kmeans.fit(X_cluster_input)
        
        # Assign cluster indices to raw data temporarily for analysis
        df_analysis = df_raw.copy()
        df_analysis["ClusterIndex"] = self.kmeans.labels_
        
        # Analyze centroids to map indices to labels
        cluster_summaries = []
        for i in range(self.n_clusters):
            cluster_data = df_analysis[df_analysis["ClusterIndex"] == i]
            summary = {
                "index": i,
                "mean_balance": cluster_data["Balance"].mean(),
                "mean_tx_vol": cluster_data["TransactionVolume"].mean(),
                "mean_complaints": cluster_data["Complaints"].mean(),
                "mean_churn": cluster_data["Churn"].mean() if "Churn" in cluster_data.columns else 0.0,
                "size": len(cluster_data)
            }
            cluster_summaries.append(summary)
            
        # Programmatically assign business labels:
        # 1. Sort by balance to find high balance vs low balance clusters
        sorted_by_balance = sorted(cluster_summaries, key=lambda x: x["mean_balance"], reverse=True)
        
        # The highest balance cluster with higher churn / complaints is "High Value At Risk"
        # The highest balance cluster with lower churn / complaints is "Loyal High Savers"
        hb_1 = sorted_by_balance[0]
        hb_2 = sorted_by_balance[1]
        
        # Compare churn or complaints between the two highest balance clusters
        if hb_1["mean_churn"] >= hb_2["mean_churn"] or hb_1["mean_complaints"] > hb_2["mean_complaints"]:
            high_val_at_risk_idx = hb_1["index"]
            loyal_savers_idx = hb_2["index"]
        else:
            high_val_at_risk_idx = hb_2["index"]
            loyal_savers_idx = hb_1["index"]
            
        # The lowest balance cluster is "Low-Value Casuals"
        low_val_casuals_idx = sorted_by_balance[-1]["index"]
        
        # The remaining cluster is "Standard Mid-Tier"
        assigned_indices = {high_val_at_risk_idx, loyal_savers_idx, low_val_casuals_idx}
        standard_mid_tier_idx = [c["index"] for c in cluster_summaries if c["index"] not in assigned_indices][0]
        
        # Create mapping
        self.cluster_mapping = {
            high_val_at_risk_idx: "High Value At Risk",
            loyal_savers_idx: "Loyal High Savers",
            low_val_casuals_idx: "Low-Value Casuals",
            standard_mid_tier_idx: "Standard Mid-Tier"
        }
        
        return self

    def predict(self, X_processed):
        """
        Predicts raw cluster index for each sample.
        """
        X_cluster_input = X_processed[self.clustering_features]
        return self.kmeans.predict(X_cluster_input)

    def get_business_segments(self, X_processed):
        """
        Predicts and maps cluster indices to business labels.
        """
        indices = self.predict(X_processed)
        return np.array([self.cluster_mapping[idx] for idx in indices])

    def save(self, filepath):
        """Saves segmenter instance using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Loads segmenter instance using joblib."""
        return joblib.load(filepath)
