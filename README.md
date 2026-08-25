# AI Pioneers: Machine Learning & Data Preprocessing Pipeline

This repository hosts the **Week 1 Project** for the **AI Pioneers Internship (Machine Learning)** at YuvaIntern. This stage of the development lifecycle focuses on building a reproducible data cleaning, engineering, and preprocessing pipeline using Python, Pandas, and NumPy to prepare messy raw data for machine learning models.

## 🚀 Project Overview

Raw, real-world data is rarely ready for machine learning algorithms. It frequently contains missing entries, incompatible data scales, and unencoded text strings. This project demonstrates an automated engineering script designed to clean these anomalies systematically, preventing mathematical bias or computation crashes during downstream model training.

### Core Objectives
* **Data Ingestion & Profiling:** Efficiently parse multi-column tables into structural DataFrames using Pandas.
* **Missing Value Imputation:** Neutralize missing profiles without shifting dataset distributions using statistical parameters (mean, median, and label markers).
* **Categorical Feature Encoding:** Map text-based categorical strings into functional numeric arrays using Ordinal Mapping and Multi-Class One-Hot Encoding.
* **Feature Scaling:** Implement Min-Max Normalization to bounded matrices (0.0 to 1.0) to eliminate high-magnitude numerical bias.
* **Data Preservation:** Process transformations without shrinking observation row counts.

---

## 🛠️ Tech Stack & Dependencies

The pipeline is written entirely in **Python 3** and leverages standard data-science ecosystem frameworks:
* **Pandas:** Used for reading, cleaning, indexing, mapping, and transforming tabular layouts.
* **NumPy:** Used for fast vectorized matrix adjustments and parsing blank data nodes (`np.nan`).

To install the necessary packages on your computer, run:
```bash
pip install pandas numpy
```

---

## 💻 Pipeline Architecture & Source Code

The pipeline contains a robust architecture that automatically generates a synthetic fallback data matrix if an external file source is missing. The logic is saved in `preprocess.py`:

```python
import pandas as pd
import numpy as np

# 1. Ingest Raw Dataset
try:
    df = pd.read_csv('employee_data.csv')
except FileNotFoundError:
    # Synthetic fallback dataframe for demonstration purposes
    np.random.seed(42)
    df = pd.DataFrame({
        'Employee_ID': [f'EMP{i:03d}' for i in range(1, 101)],
        'Age': np.random.choice([25, 30, 45, np.nan, 35, 52], size=100),
        'Annual_Salary_INR': np.random.choice([45000, 60000, np.nan, 85000, 120000], size=100),
        'Department': np.random.choice(['IT', 'HR', 'Sales', None], size=100),
        'Performance_Score': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], size=100)
    })

print("--- Step 1: Raw DataFrame Sample ---")
print(df.head(), "\n")

# 2. Mathematical Imputation of Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Annual_Salary_INR'] = df['Annual_Salary_INR'].fillna(df['Annual_Salary_INR'].mean())
df['Department'] = df['Department'].fillna('Unknown')

# 3. Categorical Transformations
performance_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
df['Performance_Numeric'] = df['Performance_Score'].map(performance_map)
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# 4. Feature Scaling (Min-Max Normalization)
min_sal = df['Annual_Salary_INR'].min()
max_sal = df['Annual_Salary_INR'].max()
df['Salary_Normalized'] = (df['Annual_Salary_INR'] - min_sal) / (max_sal - min_sal)

print("--- Step 2: Fully Cleaned and Preprocessed Data ---")
print(df.head())
print("\nPreprocessing pipeline completed successfully. Final matrix shape:", df.shape)
```

---

## 📈 Preprocessing Methodologies Documented

1. **Age Imputation via Median:** Filled empty cells using column medians. This shields the age feature from being distorted by extreme values or outliers.
2. **Salary Imputation via Mean:** Calculated average indicators across the group to provide stable baseline weights for missing income reports.
3. **Ordinal Encoding:** Converted explicit metrics (`Poor` $\rightarrow$ `Excellent`) into sequential integers (`1` $\rightarrow$ `4`) to keep order parameters intact.
4. **Nominal One-Hot Encoding:** Split nominal categorical groups (like departments) into independent binary columns. This prevents algorithms from guessing a fake mathematical relationship between categories.
5. **Min-Max Normalization:** Restructured salaries into a strict `[0.0, 1.0]` scale. This step keeps algorithms like K-Nearest Neighbors or Support Vector Machines from prioritizing massive income values over smaller variables.

---

## 📋 Deliverables Included
* `preprocess.py`: Fully operational and reproducible data engineering pipeline script.
* `Week_1_Data_Preprocessing_Report.docx`: Formal corporate evaluation report containing step-by-step methodologies and design summaries.
