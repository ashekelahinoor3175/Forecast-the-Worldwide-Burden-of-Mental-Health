
# 🌍 A Perspective on Healthcare Technologies about the Application of Sociodemographic Variables to Forecast the Worldwide Burden of Mental Health

**MSc in Computing in Healthcare Innovation and Technology**  
**Atlantic Technological University (ATU), Donegal, Ireland**

- **Author:** Ashek Elahi Noor  
- **Student ID:** L00195145  
- **Dissertation Submitted:** January 2026  
- **Supervisor:** Dr. Mara Sintejudeanu  

---

## 1. Project Overview

This repository contains the source code, processed data, trained machine learning models, and documentation supporting the MSc dissertation entitled:

**“A Perspective on Healthcare Technologies about the Application of Sociodemographic Variables to Forecast the Worldwide Burden of Mental Health.”**

The project applies supervised machine learning techniques—**Logistic Regression**, **Random Forest**, and **XGBoost**—to forecast population-level mental health risk using key sociodemographic indicators.

Core explanatory variables include:
- **Human Development Index (HDI)**
- **Disability-Adjusted Life Years (DALYs) attributable to depression**
- **Temporal trends (Year)**

The analytical objective is to classify countries into **High Depression Risk** and **Low Depression Risk** categories using a binary target variable derived from the **70th percentile** of the depression prevalence distribution.

All final results, figures, tables, and model performance metrics (accuracy, precision, recall, F1-score, ROC-AUC) are formally reported in the submitted dissertation:

`MSc_Dissertation_Final.pdf`

---

## 2. Repository Structure

The repository follows a clear, modular, and industry-aligned structure to ensure transparency, reproducibility, and academic review compliance:

| Directory / File | Contents | Purpose |
|------------------|---------|---------|
| `README.md` | This file | Project overview and replication instructions |
| `/data/` | `final_processed_data.csv` | **CRITICAL:** Final cleaned and feature-engineered dataset used for all modelling |
| `/models/` | `rf_model.joblib`, `log_reg_model.joblib`, `xgb_model.joblib`, `feature_scaler.pkl`, `model_features.pkl` | Saved trained models and preprocessing artefacts |
| `/scripts/` | Jupyter Notebook (`.ipynb`) | End-to-end preprocessing, modelling, evaluation, and visualisation pipeline |
| `/references/` | `data_documentation.txt`, `reference_list.txt` | Data documentation and academic references |

---

## 3. Replication Instructions

### A. Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/rajet3175/MSc_Mental_Health_Forecasting.git
cd MSc_Mental_Health_Forecasting
```

2. **Install Required Dependencies**

The project relies on standard Python data science libraries:
```bash
pip install pandas numpy scikit-learn matplotlib plotly joblib xgboost jupyter
```

Python **3.9 or later** is recommended.

---

### B. Model Execution and Evaluation

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Run the Main Analysis Notebook**

Open the notebook located in the `/scripts/` directory (for example:  
`Forecasting_Global_Mental_Health.ipynb`) and execute all cells sequentially.

The notebook will:
- Load the processed dataset from `/data/final_processed_data.csv`
- Apply preprocessing steps consistent with the dissertation methodology
- Load or train Logistic Regression, Random Forest, and XGBoost models
- Reproduce all evaluation metrics and figures reported in the thesis

---

## 4. Example Model Loading Code (Corrected and Reproducible)

The following Python snippet demonstrates how trained models and preprocessing artefacts stored in the `/models/` directory are loaded and used for prediction on new data.

```python
import joblib
import pandas as pd

# Paths to artefacts
RF_MODEL_PATH = 'models/rf_model.joblib'
LR_MODEL_PATH = 'models/log_reg_model.joblib'
SCALER_PATH = 'models/feature_scaler.pkl'
FEATURE_LIST_PATH = 'models/model_features.pkl'

# Load feature list
feature_columns = joblib.load(FEATURE_LIST_PATH)

# Load models and scaler
rf_model = joblib.load(RF_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Example raw input data
raw_data = pd.DataFrame({
    'HDI': [0.75],
    'DALY_Depression': [220],
    'Year': [2022],
    'HDI_Category': ['High']
})

# One-Hot Encoding
X = raw_data[['HDI', 'DALY_Depression', 'Year']].copy()
X['HDI_Cat_Low'] = (raw_data['HDI_Category'] == 'Low').astype(int)
X['HDI_Cat_Medium'] = (raw_data['HDI_Category'] == 'Medium').astype(int)
X['HDI_Cat_Very_High'] = (raw_data['HDI_Category'] == 'Very High').astype(int)

# Align columns
X = X.reindex(columns=feature_columns, fill_value=0)

# Scale numerical features
X[['HDI', 'DALY_Depression', 'Year']] = scaler.transform(
    X[['HDI', 'DALY_Depression', 'Year']]
)

# Predictions
rf_prediction = rf_model.predict(X)
lr_probability = lr_model.predict_proba(X)[:, 1]

print('Random Forest Risk Classification:', rf_prediction)
print('Logistic Regression Risk Probability:', lr_probability)
```

---

## 5. Reproducibility and Transparency Statement

This repository constitutes the authoritative reproducibility archive for the MSc dissertation. All preprocessing steps, feature engineering procedures, model configurations, and evaluation workflows are fully documented and executable.

The **High_Depression_Risk** target variable is consistently defined using the **70th percentile threshold** across all scripts, models, and reported results.

---

## 6. Academic Use and Citation

This repository is provided for academic assessment, transparency, and replication purposes. Any reuse of the code or methodology should appropriately cite the associated MSc dissertation.

---

## 7. Contact

**Ashek Elahi Noor**  
MSc in Computing in Healthcare Innovation and Technology  
Atlantic Technological University (ATU), Donegal, Ireland  
