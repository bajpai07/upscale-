import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("--- 1. Loading Unbiased Model Artifact & Data ---")
model_path = os.path.join(MODELS_DIR, "upsell_xgboost_v2.joblib")
artifact = joblib.load(model_path)
model = artifact['model']
feature_cols = artifact['feature_cols']

df = pd.read_csv(PROCESSED_DATA_PATH)

X = df[feature_cols]
y = df['Upsell_Ready_v2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Test set probabilities
y_prob = model.predict_proba(X_test)[:, 1]
df_test = X_test.copy()
df_test['Actual_Target'] = y_test
df_test['Upsell_Prob'] = y_prob
df_test['Phone_Number'] = df.loc[X_test.index, 'Phone Number']

print("\n--- 2. SHAP Explainability Analysis ---")
explainer = shap.TreeExplainer(model)
# Calculate SHAP values on test set sample (e.g. 5,000 samples for fast visualization)
sample_size = min(5000, len(X_test))
X_test_sample = X_test.iloc[:sample_size]
shap_values = explainer(X_test_sample)

# Global SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('Global SHAP Feature Summary Plot (Impact on Upsell Readiness)', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
shap_summary_path = os.path.join(PLOTS_DIR, "shap_summary.png")
plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Global SHAP plot: {shap_summary_path}")

print("\n--- 3. Individual Customer SHAP Case Studies ---")

# Customer A: High Confidence
high_idx = df_test[(df_test['Upsell_Prob'] >= 0.85) & (df_test['Actual_Target'] == 1)].index
cust_a_idx = high_idx[0] if len(high_idx) > 0 else df_test['Upsell_Prob'].idxmax()

# Customer B: Borderline / Uncertain
borderline_idx = df_test[(df_test['Upsell_Prob'] >= 0.45) & (df_test['Upsell_Prob'] <= 0.55)].index
cust_b_idx = borderline_idx[0] if len(borderline_idx) > 0 else (df_test['Upsell_Prob'] - 0.5).abs().idxmin()

# Customer C: Confidently Not Ready
low_idx = df_test[(df_test['Upsell_Prob'] <= 0.15) & (df_test['Actual_Target'] == 0)].index
cust_c_idx = low_idx[0] if len(low_idx) > 0 else df_test['Upsell_Prob'].idxmin()

customers = [
    ('High_Confidence', cust_a_idx, 'shap_customer_high.png'),
    ('Borderline_Uncertain', cust_b_idx, 'shap_customer_borderline.png'),
    ('Confidently_Not_Ready', cust_c_idx, 'shap_customer_low.png')
]

customer_summaries = []

for name, idx, plot_name in customers:
    row_feat = X_test.loc[[idx]]
    row_prob = df_test.loc[idx, 'Upsell_Prob']
    row_phone = df_test.loc[idx, 'Phone_Number']
    
    # Compute SHAP values for single instance
    sv = explainer(row_feat)
    
    # Plot waterfall plot for instance
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(sv[0], max_display=8, show=False)
    plt.title(f'SHAP Waterfall Explanation - {name.replace("_", " ")} (Phone: {row_phone}, Prob: {row_prob*100:.1f}%)', fontsize=11, fontweight='bold', pad=12)
    plot_out = os.path.join(PLOTS_DIR, plot_name)
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate Narrative
    vals = sv.values[0]
    names = feature_cols
    feat_contribs = sorted(zip(names, row_feat.values[0], vals), key=lambda x: abs(x[2]), reverse=True)
    
    top_pos = [f"{f}={v:.0f} (+{c:.3f})" for f, v, c in feat_contribs if c > 0][:2]
    top_neg = [f"{f}={v:.0f} ({c:.3f})" for f, v, c in feat_contribs if c < 0][:2]
    
    summary_text = (
        f"[{name.upper()}] Phone: {row_phone} | Predicted Prob: {row_prob*100:.1f}%\n"
        f"  - Top Drivers For Upsell: {', '.join(top_pos) if top_pos else 'None'}\n"
        f"  - Top Drivers Against Upsell: {', '.join(top_neg) if top_neg else 'None'}\n"
    )
    customer_summaries.append(summary_text)

print("\n".join(customer_summaries))

print("--- 4. Responsible AI Guardrail Evaluation ---")
# Rule: If CustServ Calls >= 4, exclude from Upsell campaign even if model prob >= 0.5

df_full = df.copy()
df_full['Model_Prob'] = model.predict_proba(df[feature_cols])[:, 1]
df_full['Model_Pred_Eligible'] = df_full['Model_Prob'] >= 0.50

# Apply Guardrail
df_full['Guardrail_Triggered'] = df_full['CustServ Calls'] >= 4
df_full['Final_Upsell_Eligible'] = df_full['Model_Pred_Eligible'] & (~df_full['Guardrail_Triggered'])

high_prob_count = df_full['Model_Pred_Eligible'].sum()
excluded_count = (df_full['Model_Pred_Eligible'] & df_full['Guardrail_Triggered']).sum()
final_eligible_count = df_full['Final_Upsell_Eligible'].sum()

print("\n--- Guardrail Exclusion Statistics ---")
print(f"Total Dataset Customers: {len(df_full):,}")
print(f"Model-Selected High Probability Customers (Prob >= 50%): {high_prob_count:,} ({high_prob_count/len(df_full)*100:.2f}%)")
print(f"Guardrail Triggered (CustServ Calls >= 4): {excluded_count:,} customers excluded")
print(f"Exclusion Rate in High-Prob Segment: {excluded_count/high_prob_count*100:.2f}%")
print(f"Final Campaign Eligible Customers (Post-Guardrail): {final_eligible_count:,} ({final_eligible_count/len(df_full)*100:.2f}%)")

# Save updated dataset with final probabilities and guardrail status
df_full.to_csv(PROCESSED_DATA_PATH, index=False)

# Save final model artifact package
final_artifact = {
    'model': model,
    'explainer': explainer,
    'feature_cols': feature_cols,
    'guardrail_threshold': {'CustServ_Calls_Max': 3},
    'metrics': artifact['metrics']
}

final_model_path = os.path.join(MODELS_DIR, "final_upsell_model.joblib")
joblib.dump(final_artifact, final_model_path)
print(f"\nSaved final model package artifact to {final_model_path}")
