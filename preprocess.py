import pandas as pd
import numpy as np

try:
    df = pd.read_csv('employee_data.csv')
except FileNotFoundError:
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

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Annual_Salary_INR'] = df['Annual_Salary_INR'].fillna(df['Annual_Salary_INR'].mean())
df['Department'] = df['Department'].fillna('Unknown')

performance_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
df['Performance_Numeric'] = df['Performance_Score'].map(performance_map)
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

min_sal = df['Annual_Salary_INR'].min()
max_sal = df['Annual_Salary_INR'].max()
df['Salary_Normalized'] = (df['Annual_Salary_INR'] - min_sal) / (max_sal - min_sal)

print("--- Step 2: Fully Cleaned and Preprocessed Data ---")
print(df.head())
print("\nPreprocessing pipeline completed successfully. Final matrix shape:", df.shape)
