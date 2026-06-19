class RetentionRecommendationEngine:
    """
    Translates model predictions, segments, and risk drivers into prioritized, actionable business playbooks.
    """
    def __init__(self):
        pass

    def generate_recommendations(self, churn_prob, segment, top_drivers):
        """
        Generates specific recommendations based on customer churn risk, segment, and SHAP risk drivers.
        
        Parameters:
        - churn_prob: float, probability of churn (0.0 to 1.0)
        - segment: str, name of the customer segment
        - top_drivers: list of dicts, from ChurnExplainer.explain_customer
        
        Returns:
        - recommendations: list of dicts containing Action, Priority, and Description.
        """
        recommendations = []
        
        # High Churn Risk (>= 30% risk threshold)
        if churn_prob >= 0.30:
            # 1. Segment-based foundation recommendations
            if segment == "High Value At Risk":
                recommendations.append({
                    "Action": "Assign Branch Relationship Manager",
                    "Priority": "CRITICAL",
                    "Description": "High balance customer with severe churn risk. Initiate phone call within 24 hours from the Branch Manager and assign a dedicated relationship manager."
                })
                recommendations.append({
                    "Action": "Offer Premium Retention Bundle",
                    "Priority": "HIGH",
                    "Description": "Waive all account fees for 12 months, upgrade credit card tier for free, and provide priority branch scheduling."
                })
            elif segment == "Standard Mid-Tier":
                recommendations.append({
                    "Action": "Proactive Loyalty Outreach",
                    "Priority": "MEDIUM",
                    "Description": "Schedule automated personal outreach by the customer success team to inquire about banking experiences and account utility."
                })
            elif segment == "Low-Value Casuals":
                recommendations.append({
                    "Action": "Product Simplification Survey",
                    "Priority": "LOW",
                    "Description": "Send brief digital survey offering to consolidate inactive products and simplify fee structures."
                })
                
            # 2. Driver-based tailored recommendations
            for driver in top_drivers:
                feat_name = driver["feature"]
                # Only offer drivers that are increasing churn risk
                if driver["direction"] != "Increase Churn":
                    continue
                    
                if feat_name == "Complaints" or feat_name == "ComplaintsPerTenure":
                    recommendations.append({
                        "Action": "Service Recovery Protocol",
                        "Priority": "CRITICAL",
                        "Description": "Customer has active complaints. Escalated case to Customer Resolution desk. Offer ₹1,500 goodwill credit / fee waiver."
                    })
                elif feat_name in ["Balance", "BalanceToSalaryRatio", "BalancePerProduct"]:
                    recommendations.append({
                        "Action": "Custom High-Yield Fixed Deposit",
                        "Priority": "HIGH",
                        "Description": "Detected declining balance. Pitch a special 12-month Fixed Deposit with +0.60% premium interest rate to secure deposit stickiness."
                    })
                elif feat_name in ["TransactionVolume", "TransactionAmount", "EngagementScore", "IsActiveMember"]:
                    recommendations.append({
                        "Action": "Card Spending Cashback Campaign",
                        "Priority": "HIGH",
                        "Description": "Detected dropping usage. Send a targeted credit/debit card offer: 5% cashback on groceries and fuel for 90 days to stimulate activity."
                    })
                elif feat_name == "NumOfProducts":
                    recommendations.append({
                        "Action": "Product Portfolio Right-Sizing",
                        "Priority": "MEDIUM",
                        "Description": "Too many or too few products. Customer is experiencing product mismatch. Offer a complimentary review session with a product advisor."
                    })

        # Low Churn Risk (< 30% risk threshold)
        else:
            if segment == "Loyal High Savers":
                recommendations.append({
                    "Action": "Enroll in Wealth Circle",
                    "Priority": "MEDIUM",
                    "Description": "High-value, stable customer. Invite to the exclusive Wealth Management tier, offering personalized investment advice and cross-sell mutual funds."
                })
                recommendations.append({
                    "Action": "Cross-Sell Premium Credit Card",
                    "Priority": "LOW",
                    "Description": "Pitch our Signature Metal credit card with airport lounge access and 2% reward points."
                })
            elif segment == "Standard Mid-Tier":
                recommendations.append({
                    "Action": "Digital Cross-Sell: Pre-Approved Loan",
                    "Priority": "LOW",
                    "Description": "Secure customer is eligible. Present a pre-approved personal loan offer with low processing fees on the mobile app."
                })
            else:
                recommendations.append({
                    "Action": "Standard Account Maintenance",
                    "Priority": "LOW",
                    "Description": "Provide standard quarterly updates, keeping the customer updated on new digital banking features."
                })

        # If no recommendation triggered, add a default fallback
        if not recommendations:
            recommendations.append({
                "Action": "Relationship Review Call",
                "Priority": "LOW",
                "Description": "Perform standard check-in callback to gauge customer satisfaction and ensure digital banking logins are operational."
            })
            
        # De-duplicate actions by title
        seen_actions = set()
        unique_recs = []
        for rec in recommendations:
            if rec["Action"] not in seen_actions:
                seen_actions.add(rec["Action"])
                unique_recs.append(rec)
                
        # Sort recommendations by priority order: CRITICAL -> HIGH -> MEDIUM -> LOW
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        unique_recs.sort(key=lambda x: priority_order.get(x["Priority"], 4))
        
        return unique_recs
