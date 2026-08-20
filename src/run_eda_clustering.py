import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Configure directories
PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "CDR-Call-Details.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "cdr_features.csv")
PLOTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "plots")
NOTEBOOK_PATH = os.path.join(PROJECT_DIR, "notebooks", "01_eda_and_persona_clustering.ipynb")

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 11

print("--- 1. Loading Raw CDR Data ---")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.info())

print("\n--- 2. Feature Engineering ---")
df['Total Mins'] = df['Day Mins'] + df['Eve Mins'] + df['Night Mins'] + df['Intl Mins']
df['Total Calls'] = df['Day Calls'] + df['Eve Calls'] + df['Night Calls'] + df['Intl Calls']
df['Total Charge'] = df['Day Charge'] + df['Eve Charge'] + df['Night Charge'] + df['Intl Charge']
df['Avg Charge per Min'] = df['Total Charge'] / (df['Total Mins'] + 1e-5)
df['CustServ Call Ratio'] = df['CustServ Calls'] / (df['Total Calls'] + 1)
df['Churn_Int'] = df['Churn'].astype(int)

print("Engineered features: Total Mins, Total Calls, Total Charge, Avg Charge per Min, CustServ Call Ratio")

print("\n--- 3. Correlation & Churn Analysis ---")
corr = df.corr(numeric_only=True)
custserv_churn_corr = corr.loc['CustServ Calls', 'Churn_Int']
print(f"Correlation between CustServ Calls and Churn: {custserv_churn_corr:.4f}")

# Group by CustServ Calls and calculate Churn Rate
cs_churn = df.groupby('CustServ Calls').agg(
    Total_Customers=('Churn_Int', 'count'),
    Churned=('Churn_Int', 'sum'),
    Churn_Rate=('Churn_Int', 'mean')
).reset_index()

print("\nCustServ Calls vs Churn Rate:")
print(cs_churn)

# Plot 1: CustServ Calls vs Churn Rate
fig, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=cs_churn, x='CustServ Calls', y='Churn_Rate', color='#e74c3c', ax=ax1, alpha=0.85)
ax1.set_title('Churn Rate by Number of Customer Service Calls', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Customer Service Calls', fontsize=12)
ax1.set_ylabel('Churn Rate', fontsize=12, color='#e74c3c')
ax1.set_ylim(0, 1.0)
for p in ax1.patches:
    height = p.get_height()
    if height > 0:
        ax1.annotate(f'{height*100:.1f}%', 
                     (p.get_x() + p.get_width() / 2., height), 
                     ha='center', va='bottom', fontsize=10, xytext=(0, 3), 
                     textcoords='offset points', fontweight='bold')

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, "churn_vs_custserv.png")
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved plot: {plot1_path}")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 10))
cols_for_corr = ['Account Length', 'VMail Message', 'Day Mins', 'Eve Mins', 'Night Mins', 
                 'Intl Mins', 'CustServ Calls', 'Total Mins', 'Total Charge', 'Churn_Int']
sns.heatmap(df[cols_for_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Saved plot: {plot2_path}")

print("\n--- 4. KMeans Persona Clustering ---")
cluster_features = [
    'Total Mins', 'Total Charge', 'Day Mins', 'Eve Mins', 'Night Mins', 
    'Intl Mins', 'CustServ Calls', 'VMail Message', 'Account Length'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features])

# Sample subset for fast silhouette calculation if dataset is large
sample_idx = np.random.choice(len(X_scaled), size=min(10000, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

k_results = {}
for k in [4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_sample, labels[sample_idx])
    inertia = km.inertia_
    k_results[k] = {'inertia': inertia, 'silhouette': sil}
    print(f"k={k}: Inertia={inertia:.2f}, Silhouette Score={sil:.4f}")

# Optimal k selection
best_k = max(k_results, key=lambda k: k_results[k]['silhouette'])
print(f"\nSelected Optimal k={best_k} based on highest Silhouette Score ({k_results[best_k]['silhouette']:.4f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Persona_Cluster'] = kmeans.fit_predict(X_scaled)

# Compute Cluster Profiles
cluster_profiles = df.groupby('Persona_Cluster').agg(
    Customer_Count=('Phone Number', 'count'),
    Percentage=('Phone Number', lambda x: f"{len(x)/len(df)*100:.1f}%"),
    Churn_Rate=('Churn_Int', 'mean'),
    Avg_Total_Mins=('Total Mins', 'mean'),
    Avg_Total_Charge=('Total Charge', 'mean'),
    Avg_Day_Mins=('Day Mins', 'mean'),
    Avg_Eve_Mins=('Eve Mins', 'mean'),
    Avg_Night_Mins=('Night Mins', 'mean'),
    Avg_Intl_Mins=('Intl Mins', 'mean'),
    Avg_CustServ_Calls=('CustServ Calls', 'mean'),
    Avg_VMail_Msg=('VMail Message', 'mean'),
    Avg_Account_Len=('Account Length', 'mean')
).reset_index()

print("\n--- Cluster Profiles (Mean Values per Persona Cluster) ---")
print(cluster_profiles.to_string(index=False))

# Plot 3: 2D PCA Plot of Clusters
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(11, 8))
# Sample 15,000 points for crisp visualization
df_plot_sample = df.sample(n=min(15000, len(df)), random_state=42)

palette = sns.color_palette("Set2", best_k)
sns.scatterplot(
    data=df_plot_sample, x='PCA1', y='PCA2', 
    hue='Persona_Cluster', palette=palette, alpha=0.5, s=25, legend='full'
)

# Plot cluster centroids in PCA space
centroids_scaled = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids_scaled)

plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1], 
    s=250, c='black', marker='X', label='Centroids', edgecolor='white', linewidth=2
)

for i, (cx, cy) in enumerate(centroids_pca):
    plt.annotate(f'Cluster {i}', (cx, cy), xytext=(cx+0.3, cy+0.3),
                 fontweight='bold', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

plt.title(f'2D PCA Projection of Customer Personas (k={best_k})', fontsize=14, fontweight='bold', pad=12)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
plt.legend(title='Persona Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, "pca_clusters.png")
plt.savefig(plot3_path, dpi=300)
plt.close()
print(f"Saved plot: {plot3_path}")

# Plot 4: Cluster Profiles Overview Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(data=cluster_profiles, x='Persona_Cluster', y='Churn_Rate', ax=axes[0, 0], palette=palette)
axes[0, 0].set_title('Churn Rate by Cluster', fontweight='bold')
axes[0, 0].set_ylabel('Churn Rate')

sns.barplot(data=cluster_profiles, x='Persona_Cluster', y='Avg_Total_Mins', ax=axes[0, 1], palette=palette)
axes[0, 1].set_title('Average Total Mins by Cluster', fontweight='bold')
axes[0, 1].set_ylabel('Total Minutes')

sns.barplot(data=cluster_profiles, x='Persona_Cluster', y='Avg_CustServ_Calls', ax=axes[1, 0], palette=palette)
axes[1, 0].set_title('Average CustServ Calls by Cluster', fontweight='bold')
axes[1, 0].set_ylabel('CustServ Calls')

sns.barplot(data=cluster_profiles, x='Persona_Cluster', y='Avg_Total_Charge', ax=axes[1, 1], palette=palette)
axes[1, 1].set_title('Average Total Charge ($) by Cluster', fontweight='bold')
axes[1, 1].set_ylabel('Total Charge ($)')

plt.suptitle('Customer Persona Profiles Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot4_path = os.path.join(PLOTS_DIR, "cluster_profiles.png")
plt.savefig(plot4_path, dpi=300)
plt.close()
print(f"Saved plot: {plot4_path}")

print("\n--- 5. Saving Processed Data ---")
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

print("\n--- 6. Generating Jupyter Notebook ---")
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 01. Exploratory Data Analysis & Customer Persona Clustering\n",
    "\n",
    "This notebook analyzes the **Telco Call Details Records (CDR)** dataset to:\n",
    "1. Perform basic Exploratory Data Analysis (EDA) on usage, charges, customer service calls, and churn rate.\n",
    "2. Evaluate correlation patterns, specifically between `CustServ Calls` and `Churn`.\n",
    "3. Engineer aggregated behavioral features (`Total Mins`, `Total Charge`, `CustServ Call Ratio`).\n",
    "4. Segment customers into distinct **Customer Personas** using KMeans clustering ($k=4..6$).\n",
    "5. Visualize customer clusters via 2D Principal Component Analysis (PCA)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Load processed dataset\n",
    "df = pd.read_csv('../data/processed/cdr_features.csv')\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Exploratory Data Analysis & Feature Distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic Summary Statistics\n",
    "summary_cols = ['Day Mins', 'Eve Mins', 'Night Mins', 'Intl Mins', 'CustServ Calls', 'VMail Message', 'Account Length', 'Total Mins', 'Total Charge', 'Churn']\n",
    "df[summary_cols].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. CustServ Calls vs Churn Rate Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cs_churn = df.groupby('CustServ Calls').agg(\n",
    "    Total_Customers=('Churn_Int', 'count'),\n",
    "    Churned=('Churn_Int', 'sum'),\n",
    "    Churn_Rate=('Churn_Int', 'mean')\n",
    ").reset_index()\n",
    "cs_churn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Customer Persona Clustering (K-Means & PCA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cluster_profiles = df.groupby('Persona_Cluster').agg(\n",
    "    Customer_Count=('Phone Number', 'count'),\n",
    "    Churn_Rate=('Churn_Int', 'mean'),\n",
    "    Avg_Total_Mins=('Total Mins', 'mean'),\n",
    "    Avg_Total_Charge=('Total Charge', 'mean'),\n",
    "    Avg_CustServ_Calls=('CustServ Calls', 'mean'),\n",
    "    Avg_VMail_Msg=('VMail Message', 'mean'),\n",
    "    Avg_Account_Len=('Account Length', 'mean')\n",
    ").reset_index()\n",
    "cluster_profiles"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook_content, f, indent=2)

print(f"Jupyter Notebook generated at {NOTEBOOK_PATH}")
print("Pipeline complete!")
