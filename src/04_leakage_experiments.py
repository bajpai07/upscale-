import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, 
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

PROJECT_DIR = r"c:\Users\bajpa\OneDrive\Desktop\Hackathon"
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

df = pd.read_csv(PROCESSED_DATA_PATH)

print("==========================================================")
print(" EXPERIMENT 1: Current Label (4 Rules) WITHOUT CustServ Calls ")
print("==========================================================")

exp1_features = [
    'Total Calls', 'Account Length', 'Intl Calls', 
    'Day Calls', 'Eve Calls', 'Night Calls', 'VMail Message'
]

X1 = df[exp1_features]
y1 = df['Upsell_Ready']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.20, random_state=42, stratify=y1
)

pos_weight1 = (len(y1_train) - sum(y1_train)) / sum(y1_train)

xgb1 = XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    scale_pos_weight=pos_weight1, random_state=42,
    eval_metric='logloss', n_jobs=-1
)
xgb1.fit(X1_train, y1_train)

y1_pred = xgb1.predict(X1_test)
y1_prob = xgb1.predict_proba(X1_test)[:, 1]

metrics1 = {
    'Precision': precision_score(y1_test, y1_pred),
    'Recall': recall_score(y1_test, y1_pred),
    'F1-Score': f1_score(y1_test, y1_pred),
    'ROC-AUC': roc_auc_score(y1_test, y1_prob),
    'PR-AUC': average_precision_score(y1_test, y1_prob)
}

fi1_df = pd.DataFrame({
    'Feature': exp1_features,
    'Importance': xgb1.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\n--- Exp 1 Feature Importances ---")
print(fi1_df.to_string(index=True))


print("\n==========================================================")
print(" EXPERIMENT 2: Unbiased Label (3 Rules, No CustServ in Label) ")
print("==========================================================")

# Build Unbiased Label: Churn == False & Total Mins >= P60 & Total Charge >= P60
p60_mins = df['Total Mins'].quantile(0.60)
p60_charge = df['Total Charge'].quantile(0.60)

c_not_churn = (df['Churn'] == False)
c_high_mins = (df['Total Mins'] >= p60_mins)
c_high_charge = (df['Total Charge'] >= p60_charge)

df['Upsell_Ready_v2'] = (c_not_churn & c_high_mins & c_high_charge).astype(int)

bal2 = df['Upsell_Ready_v2'].value_counts()
prop2 = df['Upsell_Ready_v2'].value_counts(normalize=True)

print(f"Unbiased Label (Upsell_Ready_v2) Balance:")
print(f" - Not Ready (0): {bal2.get(0,0):,} ({prop2.get(0,0)*100:.2f}%)")
print(f" - Upsell Ready (1): {bal2.get(1,0):,} ({prop2.get(1,0)*100:.2f}%)")

exp2_features = [
    'CustServ Calls', 'Total Calls', 'Account Length', 'Intl Calls', 
    'Day Calls', 'Eve Calls', 'Night Calls', 'VMail Message'
]

X2 = df[exp2_features]
y2 = df['Upsell_Ready_v2']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.20, random_state=42, stratify=y2
)

pos_weight2 = (len(y2_train) - sum(y2_train)) / sum(y2_train)

xgb2 = XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    scale_pos_weight=pos_weight2, random_state=42,
    eval_metric='logloss', n_jobs=-1
)
xgb2.fit(X2_train, y2_train)

y2_pred = xgb2.predict(X2_test)
y2_prob = xgb2.predict_proba(X2_test)[:, 1]

metrics2 = {
    'Precision': precision_score(y2_test, y2_pred),
    'Recall': recall_score(y2_test, y2_pred),
    'F1-Score': f1_score(y2_test, y2_pred),
    'ROC-AUC': roc_auc_score(y2_test, y2_prob),
    'PR-AUC': average_precision_score(y2_test, y2_prob)
}

fi2_df = pd.DataFrame({
    'Feature': exp2_features,
    'Importance': xgb2.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\n--- Exp 2 Feature Importances ---")
print(fi2_df.to_string(index=True))


print("\n==========================================================")
print("             SIDE-BY-SIDE METRICS COMPARISON              ")
print("==========================================================")

comp_metrics = pd.DataFrame({
    'Exp 1: Pure 7 Features (Original Label)': metrics1,
    'Exp 2: Unbiased Label + CustServ Predictor': metrics2
}).T

print(comp_metrics.round(4).to_string())

# Save Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
comp_metrics.plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Comparison of Leakage Experiments Performance Metrics', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
exp_plot_path = os.path.join(PLOTS_DIR, "experiments_comparison.png")
plt.savefig(exp_plot_path, dpi=300)
plt.close()
print(f"\nSaved experiments comparison plot: {exp_plot_path}")

# Update dataset with Upsell_Ready_v2
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Save Exp 2 model as candidate artifact
exp2_artifact = {
    'model': xgb2,
    'feature_cols': exp2_features,
    'metrics': metrics2,
    'exp1_metrics': metrics1,
    'label_version': 'v2_unbiased'
}
joblib.dump(exp2_artifact, os.path.join(MODELS_DIR, "upsell_xgboost_v2.joblib"))
print(f"Saved Exp 2 model to {os.path.join(MODELS_DIR, 'upsell_xgboost_v2.joblib')}")
