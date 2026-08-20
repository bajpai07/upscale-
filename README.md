# Telco Customer Upsell Classification System

An AI-driven classification and persona clustering solution for telecommunications providers. The system identifies high-potential customers for product/plan upsells based on Call Details Records (CDR) and customer care interaction behavior.

## 📌 Dataset
- **Source**: Kaggle CDR (Call Details Record) Dataset
- **URL**: [https://www.kaggle.com/datasets/anshulmehtakaggl/cdrcall-details-record-predict-telco-churn](https://www.kaggle.com/datasets/anshulmehtakaggl/cdrcall-details-record-predict-telco-churn)

---

## 📂 Project Structure

```text
Hackathon/
├── data/
│   ├── raw/          # Original raw Kaggle CDR datasets
│   └── processed/    # Cleaned, feature-engineered datasets
├── notebooks/        # Exploratory Data Analysis (EDA) & Model prototyping
├── src/              # Core python modules (preprocessing, feature engineering, training)
├── backend/          # FastAPI REST API endpoints
├── frontend/         # Streamlit interactive Web UI
├── models/           # Trained machine learning model artifacts (.joblib)
├── requirements.txt  # Project dependencies list
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
Create and activate a Python virtual environment:

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗺️ Project Roadmap
- [x] Repository scaffolding & environment setup
- [ ] Data ingestion (`kagglehub`) & Exploratory Data Analysis (EDA)
- [ ] Persona clustering & feature engineering pipeline
- [ ] Model training, evaluation & SHAP explainability analysis
- [ ] FastAPI backend service development
- [ ] Streamlit interactive dashboard frontend
- [ ] Live deployment
