import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, 
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("==================================================")
print(" RETRAINING MODEL ON DEDUPLICATED CLEAN DATASET ")
print("==================================================")

df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"Cleaned Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Re-engineer quantiles and unbiased target on clean data
p60_mins = df['Total Mins'].quantile(0.60)
p60_charge = df['Total Charge'].quantile(0.60)

c_not_churn = (df['Churn'] == False)
c_high_mins = (df['Total Mins'] >= p60_mins)
c_high_charge = (df['Total Charge'] >= p60_charge)

df['Upsell_Ready_v2'] = (c_not_churn & c_high_mins & c_high_charge).astype(int)

target_balance = df['Upsell_Ready_v2'].value_counts()
target_prop = df['Upsell_Ready_v2'].value_counts(normalize=True)

print("\n--- Clean Dataset Target Label Balance (Upsell_Ready_v2) ---")
print(f" - Not Ready (0): {target_balance.get(0,0):,} ({target_prop.get(0,0)*100:.2f}%)")
print(f" - Upsell Ready (1): {target_balance.get(1,0):,} ({target_prop.get(1,0)*100:.2f}%)")

# 8 Leakage-Free Features
honest_features = [
    'Account Length', 
    'VMail Message', 
    'CustServ Calls', 
    'Intl Calls', 
    'Day Calls', 
    'Eve Calls', 
    'Night Calls', 
    'Total Calls'
]

X = df[honest_features]
y = df['Upsell_Ready_v2']

print(f"\n--- Stratified 80/20 Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,} rows (Positives: {sum(y_train):,}, {y_train.mean()*100:.2f}%)")
print(f"Test size:  {len(X_test):,} rows (Positives: {sum(y_test):,}, {y_test.mean()*100:.2f}%)")

pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# --- 1. Baseline Model: Logistic Regression ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_metrics = {
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr),
    'ROC-AUC': roc_auc_score(y_test, y_prob_lr),
    'PR-AUC': average_precision_score(y_test, y_prob_lr)
}

# --- 2. Main Model: XGBoost Classifier ---
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_metrics = {
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1-Score': f1_score(y_test, y_pred_xgb),
    'ROC-AUC': roc_auc_score(y_test, y_prob_xgb),
    'PR-AUC': average_precision_score(y_test, y_prob_xgb)
}

print("\n==================================================")
print("   CLEAN DATASET MODEL PERFORMANCE METRICS        ")
print("==================================================")
comparison_df = pd.DataFrame({
    'Logistic Regression (Baseline)': lr_metrics,
    'XGBoost Classifier (Main Model)': xgb_metrics
}).T

print(comparison_df.round(4).to_string())

print("\n--- Clean XGBoost Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN):  {tn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"False Negatives (FN): {fn:,}")
print(f"True Positives (TP):  {tp:,}")

plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Ready (0)', 'Upsell Ready (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Clean XGBoost Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot: {cm_plot_path}")

print("\n--- Clean XGBoost Feature Importances ---")
importances = xgb_model.feature_importances_
fi_df = pd.DataFrame({
    'Feature': honest_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(fi_df.to_string(index=True))

plt.figure(figsize=(9, 5))
sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
plt.title('Clean XGBoost Feature Importances (Leakage-Free & Deduplicated)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('XGBoost Gain Importance', fontsize=11)
plt.tight_layout()
fi_plot_path = os.path.join(PLOTS_DIR, "feature_importances.png")
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"Saved feature importances plot: {fi_plot_path}")

# Initialize SHAP explainer for final artifact
import shap
explainer = shap.TreeExplainer(xgb_model)

# Save updated dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Save final model artifact package
final_artifact = {
    'model': xgb_model,
    'explainer': explainer,
    'scaler': scaler,
    'feature_cols': honest_features,
    'guardrail_threshold': {'CustServ_Calls_Max': 3},
    'metrics': xgb_metrics,
    'lr_metrics': lr_metrics
}

final_model_path = os.path.join(MODELS_DIR, "final_upsell_model.joblib")
joblib.dump(final_artifact, final_model_path)
print(f"\nSaved updated clean model artifact package to {final_model_path}")
