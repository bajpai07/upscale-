import os
import pandas as pd
import numpy as np

PROJECT_DIR = r"c:\Users\bajpa\OneDrive\Desktop\Hackathon"
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "CDR-Call-Details.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")

print("==================================================")
print(" PHASE 1: DATA AUDIT ON RAW DATASET ")
print("==================================================")

df_raw = pd.read_csv(RAW_DATA_PATH)
total_raw_rows = len(df_raw)
print(f"Raw Input Dataset: {total_raw_rows:,} rows, {df_raw.shape[1]} columns\n")

# 1. Check Missing / Null / Blank values
null_counts = df_raw.isnull().sum()
total_nulls = null_counts.sum()
print(f"1. Missing / Null / Blank Values: {total_nulls:,} total across all columns")

# 2. Check Negative Values in Numeric Columns
numeric_cols = [
    'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
    'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 
    'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls'
]

neg_rows_mask = (df_raw[numeric_cols] < 0).any(axis=1)
neg_count = neg_rows_mask.sum()
print(f"2. Negative Values in Numeric Columns: {neg_count:,} rows ({neg_count/total_raw_rows*100:.2f}%)")

if neg_count > 0:
    print("\n--- 5 Example Rows with Negative Values ---")
    print(df_raw[neg_rows_mask].head(5).to_string())

# 3. Check Extreme Synthetic / Corrupted Outliers
# Rules for extreme synthetic corruption:
# - Account Length > 1,000 days (~2.7 years)
# - VMail Message > 200 messages
# - CustServ Calls > 20 calls
# - Day Mins / Eve Mins / Night Mins > 2,000 minutes
# - Day Calls / Eve Calls / Night Calls > 500 calls

outlier_account = df_raw['Account Length'] > 1000
outlier_vmail = df_raw['VMail Message'] > 200
outlier_custserv = df_raw['CustServ Calls'] > 20
outlier_mins = (df_raw['Day Mins'] > 2000) | (df_raw['Eve Mins'] > 2000) | (df_raw['Night Mins'] > 2000)
outlier_calls = (df_raw['Day Calls'] > 500) | (df_raw['Eve Calls'] > 500) | (df_raw['Night Calls'] > 500)

all_outliers_mask = outlier_account | outlier_vmail | outlier_custserv | outlier_mins | outlier_calls
outliers_count = all_outliers_mask.sum()

print(f"\n3. Extreme Synthetic / Corrupted Outliers:")
print(f"   - Account Length > 1,000 days : {outlier_account.sum():,} rows")
print(f"   - VMail Message > 200 msgs    : {outlier_vmail.sum():,} rows")
print(f"   - CustServ Calls > 20 calls   : {outlier_custserv.sum():,} rows")
print(f"   - Call Duration > 2,000 mins  : {outlier_mins.sum():,} rows")
print(f"   - Call Counts > 500 calls     : {outlier_calls.sum():,} rows")
print(f"   - TOTAL Corrupted Rows        : {outliers_count:,} ({outliers_count/total_raw_rows*100:.2f}%)")

if outliers_count > 0:
    print("\n--- 5 Example Rows with Extreme Corrupted Outliers ---")
    print(df_raw[all_outliers_mask].head(5).to_string())


print("\n==================================================")
print(" PHASE 2: 2-STAGE CLEANING PIPELINE EXECUTION ")
print("==================================================")

# STAGE 1: Deduplication by Unique Phone Number (Keep First)
print("--- STAGE 1: Deduplication by Unique Customer (Phone Number) ---")
df_dedup = df_raw.drop_duplicates(subset=['Phone Number'], keep='first').copy().reset_index(drop=True)
dedup_rows = len(df_dedup)
removed_dups = total_raw_rows - dedup_rows

print(f"Raw Input Rows           : {total_raw_rows:,}")
print(f"Duplicate Rows Removed   : {removed_dups:,} ({removed_dups/total_raw_rows*100:.2f}%)")
print(f"Deduplicated Rows        : {dedup_rows:,} unique customers")

# STAGE 2: Handling Invalid Values & Domain Capping on Deduplicated Data
print("\n--- STAGE 2: Handling Invalid Values & Capping Outliers ---")

# Rule A: Filter out invalid negative or missing rows if present
valid_mask = ~(df_dedup[numeric_cols] < 0).any(axis=1) & df_dedup['Phone Number'].notnull()
df_clean = df_dedup[valid_mask].copy().reset_index(drop=True)
filtered_invalid = len(df_dedup) - len(df_clean)
print(f"Filtered Invalid/Negative Rows: {filtered_invalid:,}")

# Rule B: Apply Domain Capping to extreme synthetic outliers (rather than dropping valid customers)
# - Cap Account Length at 1,000 days (~2.7 years)
# - Cap VMail Message at 100 msgs
# - Cap CustServ Calls at 10 calls
# - Cap Day Calls, Eve Calls, Night Calls at 400 calls

cap_acc = (df_clean['Account Length'] > 1000).sum()
cap_vm = (df_clean['VMail Message'] > 100).sum()
cap_cs = (df_clean['CustServ Calls'] > 10).sum()
cap_day_c = (df_clean['Day Calls'] > 400).sum()
cap_eve_c = (df_clean['Eve Calls'] > 400).sum()
cap_night_c = (df_clean['Night Calls'] > 400).sum()

df_clean['Account Length'] = df_clean['Account Length'].clip(upper=1000)
df_clean['VMail Message'] = df_clean['VMail Message'].clip(upper=100)
df_clean['CustServ Calls'] = df_clean['CustServ Calls'].clip(upper=10)
df_clean['Day Calls'] = df_clean['Day Calls'].clip(upper=400)
df_clean['Eve Calls'] = df_clean['Eve Calls'].clip(upper=400)
df_clean['Night Calls'] = df_clean['Night Calls'].clip(upper=400)
df_clean['Intl Calls'] = df_clean['Intl Calls'].clip(upper=50)

print(f"\nApplied Domain Capping Statistics:")
print(f" - Account Length > 1,000 days capped  : {cap_acc:,} rows")
print(f" - VMail Message > 100 capped         : {cap_vm:,} rows")
print(f" - CustServ Calls > 10 capped          : {cap_cs:,} rows")
print(f" - Day Calls > 400 capped              : {cap_day_c:,} rows")
print(f" - Eve Calls > 400 capped              : {cap_eve_c:,} rows")
print(f" - Night Calls > 400 capped            : {cap_night_c:,} rows")

print(f"\nFINAL CLEAN DATASET ROW COUNT: {len(df_clean):,} unique clean customers")

# Feature Engineering
df_clean['Total Mins'] = df_clean['Day Mins'] + df_clean['Eve Mins'] + df_clean['Night Mins'] + df_clean['Intl Mins']
df_clean['Total Calls'] = df_clean['Day Calls'] + df_clean['Eve Calls'] + df_clean['Night Calls'] + df_clean['Intl Calls']
df_clean['Total Charge'] = df_clean['Day Charge'] + df_clean['Eve Charge'] + df_clean['Night Charge'] + df_clean['Intl Charge']
df_clean['Avg Charge per Min'] = df_clean['Total Charge'] / (df_clean['Total Mins'] + 1e-5)
df_clean['CustServ Call Ratio'] = df_clean['CustServ Calls'] / (df_clean['Total Calls'] + 1)
df_clean['Churn_Int'] = df_clean['Churn'].astype(int)

# Export cleaned dataset
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nSaved cleaned dataset to {PROCESSED_DATA_PATH}")
