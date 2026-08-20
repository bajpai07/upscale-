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

# Configure directory paths
PROJECT_DIR = r"c:\Users\bajpa\OneDrive\Desktop\Hackathon"
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(PROCESSED_DATA_PATH)

print("==================================================")
print(" 1. FULL FEATURE IMPORTANCES (PREVIOUS LEAKED MODEL) ")
print("==================================================")
all_leakage_features = [
    'Account Length', 'VMail Message', 
    'Day Mins', 'Day Calls', 'Day Charge', 
    'Eve Mins', 'Eve Calls', 'Eve Charge', 
    'Night Mins', 'Night Calls', 'Night Charge', 
    'Intl Mins', 'Intl Calls', 'Intl Charge', 
    'CustServ Calls', 'Total Mins', 'Total Calls', 
    'Total Charge', 'Avg Charge per Min', 'CustServ Call Ratio'
]

# Quick load previous model if available to print full ranking
old_model_path = os.path.join(MODELS_DIR, "upsell_xgboost_model.joblib")
if os.path.exists(old_model_path):
    old_artifact = joblib.load(old_model_path)
    old_xgb = old_artifact['model']
    old_fi = pd.DataFrame({
        'Feature': all_leakage_features,
        'Importance': old_xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    print(old_fi.to_string(index=True))

print("\n==================================================")
print(" 2. RETRAINING HONEST MODEL (LEAKAGE-FREE FEATURES)")
print("==================================================")

# Clean, non-leaked feature list
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

print(f"Honest Feature Subset ({len(honest_features)} features):")
print(honest_features)

X = df[honest_features]
y = df['Upsell_Ready']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# --- Logistic Regression Baseline ---
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

# --- XGBoost Honest Model ---
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

print("\n--- HONEST MODEL PERFORMANCE METRICS ---")
comparison_df = pd.DataFrame({
    'Logistic Regression (Baseline)': lr_metrics,
    'XGBoost (Honest Model)': xgb_metrics
}).T

print(comparison_df.round(4).to_string())

print("\n--- HONEST XGBoost CONFUSION MATRIX ---")
cm = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN):  {tn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"False Negatives (FN): {fn:,}")
print(f"True Positives (TP):  {tp:,}")

plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Ready (0)', 'Upsell Ready (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Honest XGBoost Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"Saved honest confusion matrix plot: {cm_plot_path}")

print("\n--- HONEST FEATURE IMPORTANCE RANKING ---")
importances = xgb_model.feature_importances_
honest_fi_df = pd.DataFrame({
    'Feature': honest_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(honest_fi_df.to_string(index=True))

# Plot Honest Feature Importances
plt.figure(figsize=(9, 5))
sns.barplot(data=honest_fi_df, x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
plt.title('Honest XGBoost Feature Importances (Leakage-Free)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('XGBoost Gain Importance', fontsize=11)
plt.tight_layout()
fi_plot_path = os.path.join(PLOTS_DIR, "feature_importances.png")
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"Saved honest feature importances plot: {fi_plot_path}")

print("\n--- Saving Honest Model Artifact ---")
model_artifact = {
    'model': xgb_model,
    'scaler': scaler,
    'feature_cols': honest_features,
    'metrics': xgb_metrics,
    'lr_metrics': lr_metrics
}

model_save_path = os.path.join(MODELS_DIR, "upsell_xgboost_model.joblib")
joblib.dump(model_artifact, model_save_path)
print(f"Successfully saved honest trained model pipeline to {model_save_path}")
