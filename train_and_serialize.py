import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Generate Capstone Dataset
np.random.seed(42)
X = np.random.uniform(low=[1, 40, 0], high=[15, 100, 5], size=(300, 3))
y = (0.4 * X[:, 0] + 0.2 * X[:, 1] + 1.5 * X[:, 2] > 22).astype(int)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and Fit Pipeline Components
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Serialize Model Artifacts using Joblib
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(model, 'model.joblib')

print("--- Step 1: Model Training & Serialization Complete ---")
print("Saved artifacts: 'scaler.joblib' and 'model.joblib'")
