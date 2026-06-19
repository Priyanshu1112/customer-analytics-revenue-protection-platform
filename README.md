# Customer Analytics & Revenue Protection Platform

An end-to-end, portfolio-quality customer analytics and retention system for banking institutions. This platform uses machine learning and financial models to predict customer churn, segment cohorts, estimate Customer Lifetime Value (CLV), quantify revenue at risk, explain individual drivers using SHAP, and automatically trigger retention playbooks.

---

## 📂 Project Structure

```
customer-analytics-revenue-protection-platform/
├── data/
│   └── customer_churn_data.csv        # Auto-generated 5,000 customer banking dataset
├── notebooks/
│   └── exploration.ipynb              # Walkthrough of EDA, training, and explanations
├── src/
│   ├── preprocess.py                  # Synthetic data generator & feature pipeline
│   ├── train.py                       # XGBoost & KMeans training & profiling scripts
│   ├── predict.py                     # Unified inference pipeline API for client requests
│   ├── explainability.py              # SHAP TreeExplainer local & global visualizers
│   ├── segmentation.py                # KMeans wrapper with dynamic label mapping
│   ├── clv.py                         # NIM and fee-based Customer Lifetime Value model
│   ├── revenue_risk.py                # Individual & portfolio Revenue at Risk estimators
│   ├── dashboard_metrics.py           # Dashboard compiler & visual report generator
│   └── recommendation_engine.py       # Rule-based playbook generator mapping SHAP to playbooks
├── models/
│   ├── preprocessor.joblib            # Saved preprocess scaling & category fits
│   ├── segmenter.joblib               # Saved K-Means customer segmenter
│   └── churn_model.joblib             # Trained XGBoost classifier
├── screenshots/
│   ├── shap_summary.png               # Global beeswarm feature importance
│   ├── shap_waterfall_cust_99001.png  # High-risk customer local waterfall explanation
│   └── executive_dashboard.png        # Four-panel executive business dashboard
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Build, cache, model weights, and raw data ignores
├── verify.py                          # Complete end-to-end script to run the platform
└── README.md                          # Platform documentation (this file)
```

---

## 💼 Business Problem
Retaining existing retail banking customers is significantly cheaper than acquiring new ones. Furthermore, in high-value segments (e.g., Wealth Management), individual customer attrition can directly impact deposit bases and fee income. 

Traditional churn models notify risk managers of *who* is likely to leave, but they fail to:
1. **Quantify the financial exposure**: A high-risk customer with ₹10,000 balance is less critical than one with ₹2,000,000.
2. **Explain the root cause**: Standard black-box models do not explain why a specific customer is leaving.
3. **Recommend intervention**: Predictive outputs are rarely paired with automated, personalized marketing or service recovery plans.

This platform bridges these gaps by combining **Predictive ML (XGBoost)**, **Unsupervised Segmentation (K-Means)**, **Explainable AI (SHAP)**, and **Financial Valuations (CLV & Revenue Risk)**.

---

## 🧮 Mathematical & Machine Learning Concepts

### 1. Classification Concepts (XGBoost)
* **XGBoost Classifier**: Extreme Gradient Boosting builds an ensemble of weak decision trees sequentially. It optimizes a log-loss objective function by computing gradients and hessians of the loss.
* **Class Imbalance & `scale_pos_weight`**: Banking datasets typically suffer from class imbalance (e.g., ~13% churners). Without correction, classifiers favor the majority class. We use `scale_pos_weight = negative_samples / positive_samples` to scale the loss of the positive class, ensuring balanced precision and recall.
* **ROC-AUC & PR-AUC**:
  * **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. It is robust to label distributions.
  * **PR-AUC (Precision-Recall Area Under Curve)**: Focuses heavily on the positive class (churners) and measures the trade-off between precision (reducing false alarms) and recall (capturing true churners). It is highly informative for imbalanced data.

### 2. Customer Segmentation (K-Means)
Customers are clustered using **K-Means clustering** over continuous financial variables:
$$\min_{\mathbf{S}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
* **Dynamic Centroid Profiling**: Standard K-Means yields numeric cluster tags (`0, 1, 2, 3`). We programmatically map these clusters to business personas by sorting centroids on **Balance** and **Complaints / Churn Risk**:
  1. *High Balance + High Churn Risk* $\rightarrow$ **High Value At Risk**
  2. *High Balance + Low Churn Risk* $\rightarrow$ **Loyal High Savers**
  3. *Low Balance + Low Transaction Volume* $\rightarrow$ **Low-Value Casuals**
  4. *Moderate Balance + Moderate Volume* $\rightarrow$ **Standard Mid-Tier**

### 3. Feature Engineering
We construct indicators from raw banking variables:
* **Balance-to-Salary Ratio**: $\frac{\text{Balance}}{\text{Estimated Salary}}$ (represents wealth accumulation in the bank).
* **Average Transaction Size**: $\frac{\text{Transaction Amount}}{\text{Transaction Volume}}$ (differentiates high-frequency small-ticket spenders from low-frequency high-ticket spenders).
* **Complaints Per Tenure Month**: $\frac{\text{Complaints}}{\text{Tenure Months} + 1}$ (measures immediate customer frustration scaling by their relationship length).
* **Engagement Score**: A composite index (0 to 6) representing product count, active member flags, and transaction frequency.

### 4. Customer Lifetime Value (CLV)
We employ a perpetual valuation model adjusted for risk:
$$\text{Annual Profit} = (\text{Balance} \times \text{NIM}) + (\text{Transaction Fees}) + (\text{Cross-Sell Income}) - (\text{Operating Cost})$$
$$\text{CLV} = \frac{\text{Annual Profit}}{\text{Predicted Churn Probability} + \text{Discount Rate}}$$
* *Net Interest Margin (NIM)* is set at 3% of the customer's balance.
* *Transaction Fees* are set at ₹10 per transaction plus 0.1% of volume.
* *Discount Rate* is set at 10% (bank's cost of capital).
* This ensures that as predicted churn risk approaches 100%, the denominator increases, and the CLV decreases, modeling the expected shortening of the relationship.

### 5. Revenue At Risk
$$\text{Revenue At Risk} = \text{Predicted Churn Probability} \times \text{CLV}$$
This measures the mathematical expectation of future margin loss.

### 6. Explainable AI (SHAP)
We apply **SHAP (SHapley Additive exPlanations)** based on game theory to explain individual customer scores:
* The prediction is decomposed into sum of feature attributions:
  $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
  where $\phi_0$ is the base expected log-odds of churn, and $\phi_i$ is the contribution of feature $i$.
* **Local Explanations**: Waterfall plots visualize feature attributions pushing individual customer probabilities up or down relative to the baseline.
* **Global Explanations**: Summary beeswarm plots visualize feature densities vs. their impact, exposing overall drivers (like how complaints always push churn risk upwards).

---

## 🏦 Banking Use Cases & Actionable Playbooks

The **Retention Recommendation Engine** maps predictions and SHAP drivers directly to business playbooks:

### 1. High-Value At-Risk Recovery
* **Condition**: Customer is in `High Value At Risk` segment and Churn Risk $\ge 30\%$.
* **Playbook**: Assign a Senior Branch Relationship Manager, schedule an immediate direct check-in call, and offer a dedicated VIP fee waiver bundle.

### 2. Service Recovery for Complaints
* **Condition**: Churn Risk $\ge 30\%$ and the top positive SHAP driver is `Complaints`.
* **Playbook**: Trigger the Service Recovery Protocol, escalating the case to the Branch Resolution desk, and grant ₹1,500 in compensatory reward points.

### 3. Yield Pitch for Slipping Balance
* **Condition**: Churn Risk $\ge 30\%$ and the top positive SHAP driver is `Balance` or `BalanceToSalaryRatio` (indicating declining deposits).
* **Playbook**: Offer a specialized high-yield fixed deposit promotion (+0.60% premium rate) to lock in the customer's capital.

---

## 🚀 Execution & Quickstart

### Prerequisites
* Python 3.11+
* Command prompt (PowerShell or Bash)

### Installation
1. Clone or copy the folder:
   ```bash
   cd customer-analytics-revenue-protection-platform
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the End-to-End Pipeline
To generate the dataset, train models, output predictions, compile reports, and write visual screenshots, execute:
```bash
python verify.py
```

### Script Manual Execution
* **Generate Data and fit Preprocessor**:
  ```bash
  python src/preprocess.py
  ```
* **Train & Evaluate Models**:
  ```bash
  python src/train.py
  ```
* **Predict single profile details**:
  ```bash
  python src/predict.py
  ```
* **Generate Executive Dashboard Visuals**:
  ```bash
  python src/dashboard_metrics.py
  ```

---

## 📈 Business Impact & Performance Summary

* **Valuation Saved**: The platform isolates the `High Value At Risk` cohort (representing only 16.5% of the customer base but carrying ₹4.41M in exposure).
* **High-Touch Prioritization**: Rather than cold-calling all customers, branch relationship managers focus their efforts on the top 100 high-risk, high-CLV accounts, reducing customer acquisition costs and protecting bank margins.

## 🛠️ Future Improvements
1. **Dynamic Churn Thresholds**: Adapt the 30% risk cutoff dynamically per segment based on marginal customer recovery costs.
2. **Transaction Time Series**: Integrate recurrent neural networks (LSTM) or transformer models to identify sudden shifts in transaction velocity.
3. **Automated Campaign A/B Testing**: Connect the recommendation engine output to a marketing pipeline to track recovery rates for each playbook.
