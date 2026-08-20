import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser(description="Telco Customer Upsell Classification Pipeline Demo")
    parser.add_argument("--data", type=str, default="data/raw/CDR-Call-Details.csv", help="Path to raw CDR CSV dataset")
    args = parser.parse_args()

    print("==================================================")
    print("      TELCO CUSTOMER UPSELL CLASSIFICATION PIPELINE ")
    print("==================================================")

    if not os.path.exists(args.data):
        print(f"Error: Data file not found at '{args.data}'")
        sys.exit(1)

    print(f"\n--- STEP 1: Ingesting & Deduplicating Raw Data from '{args.data}' ---")
    raw_df = pd.read_csv(args.data)
    initial_shape = raw_df.shape
    print(f"Raw Input Dataset: {initial_shape[0]:,} rows, {initial_shape[1]} columns")

    # Audit & remove duplicates
    dups_count = raw_df.duplicated().sum()
    df = raw_df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {dups_count:,} exact duplicate rows ({dups_count/initial_shape[0]*100:.2f}% of dataset)")
    print(f"Cleaned Working Dataset: {len(df):,} rows")

    print("\n--- STEP 2: Feature Engineering & Target Label Construction ---")
    df['Total Mins'] = df['Day Mins'] + df['Eve Mins'] + df['Night Mins'] + df['Intl Mins']
    df['Total Calls'] = df['Day Calls'] + df['Eve Calls'] + df['Night Calls'] + df['Intl Calls']
    df['Total Charge'] = df['Day Charge'] + df['Eve Charge'] + df['Night Charge'] + df['Intl Charge']

    p60_mins = df['Total Mins'].quantile(0.60)
    p60_charge = df['Total Charge'].quantile(0.60)

    # Unbiased Target: Active + High Mins + High Charge (No CustServ Calls in target definition)
    c_not_churn = (df['Churn'] == False)
    c_high_mins = (df['Total Mins'] >= p60_mins)
    c_high_charge = (df['Total Charge'] >= p60_charge)

    df['Upsell_Ready'] = (c_not_churn & c_high_mins & c_high_charge).astype(int)

    pos_count = df['Upsell_Ready'].sum()
    pos_pct = pos_count / len(df) * 100
    print(f"Target Label Balance ('Upsell_Ready'): {pos_count:,} positives out of {len(df):,} ({pos_pct:.2f}%)")

    # Non-leaked features
    feature_cols = [
        'CustServ Calls', 'Total Calls', 'Account Length', 'Intl Calls', 
        'Day Calls', 'Eve Calls', 'Night Calls', 'VMail Message'
    ]

    X = df[feature_cols]
    y = df['Upsell_Ready']

    print("\n--- STEP 3: Stratified Train/Test Split & Model Training ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # Baseline: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # Main: XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score

    print("\n--- MODEL PERFORMANCE METRICS (TEST SET) ---")
    metrics_df = pd.DataFrame({
        'Logistic Regression (Baseline)': {
            'Accuracy': accuracy_score(y_test, y_pred_lr),
            'Precision': precision_score(y_test, y_pred_lr),
            'Recall': recall_score(y_test, y_pred_lr),
            'F1-Score': f1_score(y_test, y_pred_lr),
            'ROC-AUC': roc_auc_score(y_test, y_prob_lr),
            'PR-AUC': average_precision_score(y_test, y_prob_lr)
        },
        'XGBoost Classifier (Main Model)': {
            'Accuracy': accuracy_score(y_test, y_pred_xgb),
            'Precision': precision_score(y_test, y_pred_xgb),
            'Recall': recall_score(y_test, y_pred_xgb),
            'F1-Score': f1_score(y_test, y_pred_xgb),
            'ROC-AUC': roc_auc_score(y_test, y_prob_xgb),
            'PR-AUC': average_precision_score(y_test, y_prob_xgb)
        }
    }).T
    print(metrics_df.round(4).to_string())

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    tn, fp, fn, tp = cm_xgb.ravel()
    print(f"\nXGBoost Confusion Matrix (Test Set: {len(y_test):,} rows):")
    print(f" - True Negatives (TN)  : {tn:,}")
    print(f" - False Positives (FP) : {fp:,}")
    print(f" - False Negatives (FN) : {fn:,}")
    print(f" - True Positives (TP)  : {tp:,}")

    print("\n--- STEP 4: SHAP Explainability & Feature Importances ---")
    explainer = shap.TreeExplainer(xgb)
    importances = xgb.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    print("XGBoost Feature Importance Ranking:")
    print(fi_df.to_string(index=True))

    print("\n--- STEP 5: Responsible AI Guardrail Evaluation ---")
    df['Model_Prob'] = xgb.predict_proba(df[feature_cols])[:, 1]
    df['Raw_Eligible'] = df['Model_Prob'] >= 0.50
    df['Guardrail_Triggered'] = df['CustServ Calls'] >= 4
    df['Final_Eligible'] = df['Raw_Eligible'] & (~df['Guardrail_Triggered'])

    high_prob_cnt = df['Raw_Eligible'].sum()
    excluded_cnt = (df['Raw_Eligible'] & df['Guardrail_Triggered']).sum()
    final_cnt = df['Final_Eligible'].sum()

    print(f"Model-Selected High Probability Customers (Prob >= 50%): {high_prob_cnt:,} ({high_prob_cnt/len(df)*100:.2f}%)")
    print(f"Guardrail Exclusions (CustServ Calls >= 4): {excluded_cnt:,} customers excluded ({excluded_cnt/high_prob_cnt*100:.2f}% of high-prob pool)")
    print(f"Final Campaign Eligible Customers: {final_cnt:,} ({final_cnt/len(df)*100:.2f}%)")

    print("\n--- STEP 6: Saving Processed Artifacts & Model Package ---")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/cdr_features.csv", index=False)
    
    final_artifact = {
        'model': xgb,
        'explainer': explainer,
        'feature_cols': feature_cols,
        'guardrail_threshold': {'CustServ_Calls_Max': 3},
        'metrics': metrics_df.to_dict(orient='index')
    }
    model_save_path = "models/final_upsell_model.joblib"
    joblib.dump(final_artifact, model_save_path)
    print(f"Saved pre-trained model package to '{model_save_path}'")
    print(f"Saved processed dataset to 'data/processed/cdr_features.csv'")

    print("\nSUCCESS: PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
