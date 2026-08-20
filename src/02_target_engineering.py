import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

print("--- STEP 1: Building Composite Target Label (Upsell_Ready) ---")
df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Compute 60th percentiles
p60_mins = df['Total Mins'].quantile(0.60)
p60_charge = df['Total Charge'].quantile(0.60)

print(f"\nRule Thresholds:")
print(f" - Not Churned (Churn == False)")
print(f" - High Engagement: Total Mins >= 60th Percentile ({p60_mins:.2f} mins)")
print(f" - Low Friction: CustServ Calls <= 1")
print(f" - High Spending Headroom: Total Charge >= 60th Percentile (${p60_charge:.2f})")

# Sub-conditions
c_not_churn = (df['Churn'] == False)
c_high_mins = (df['Total Mins'] >= p60_mins)
c_low_custserv = (df['CustServ Calls'] <= 1)
c_high_charge = (df['Total Charge'] >= p60_charge)

# Composite binary label
df['Upsell_Ready'] = (c_not_churn & c_high_mins & c_low_custserv & c_high_charge).astype(int)

# Class balance
balance = df['Upsell_Ready'].value_counts()
prop = df['Upsell_Ready'].value_counts(normalize=True)

print("\n--- Upsell_Ready Class Balance ---")
for val in [0, 1]:
    cnt = balance.get(val, 0)
    pct = prop.get(val, 0) * 100
    label_str = "Upsell Ready (1)" if val == 1 else "Not Ready (0)"
    print(f"  {label_str}: {cnt:,} ({pct:.2f}%)")

print("\n--- STEP 2: Validating Label Against Customer Personas ---")
cluster_upsell = df.groupby('Persona_Cluster').agg(
    Total_Customers=('Phone Number', 'count'),
    Upsell_Ready_Count=('Upsell_Ready', 'sum'),
    Upsell_Ready_Pct=('Upsell_Ready', lambda x: x.mean() * 100)
).reset_index()

# Add persona descriptive labels from EDA
persona_names = {
    0: "Cluster 0: Standard Mass Market",
    1: "Cluster 1: Eve/Night Heavy Power",
    2: "Cluster 2: Voicemail Power Users",
    3: "Cluster 3: High-Tenure Veterans",
    4: "Cluster 4: Day Usage High Spenders"
}
cluster_upsell['Persona_Name'] = cluster_upsell['Persona_Cluster'].map(persona_names)

print("\nUpsell_Ready Concentration by Persona Cluster:")
print(cluster_upsell[['Persona_Cluster', 'Persona_Name', 'Total_Customers', 'Upsell_Ready_Count', 'Upsell_Ready_Pct']].to_string(index=False))

# Save Validation Bar Chart
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set2", len(cluster_upsell))
ax = sns.barplot(data=cluster_upsell, x='Persona_Cluster', y='Upsell_Ready_Pct', palette=palette)

plt.title('Validation: % Upsell_Ready Customers by Persona Cluster', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Persona Cluster ID', fontsize=12)
plt.ylabel('% Upsell_Ready', fontsize=12)
plt.ylim(0, max(cluster_upsell['Upsell_Ready_Pct']) * 1.25)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=11, xytext=(0, 4), 
                textcoords='offset points', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "upsell_by_cluster.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"\nSaved validation plot: {plot_path}")

# Save updated dataframe
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Updated dataset with target label saved to {PROCESSED_DATA_PATH}")
