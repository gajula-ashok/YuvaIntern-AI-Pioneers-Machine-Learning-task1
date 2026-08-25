import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Generate Synthetic Classification Dataset
print("--- Step 1: Synthesizing Supervised Data Matrix ---")
np.random.seed(42)
n_samples = 200

# Features: Experience (years), Test_Score (0-100), Certifications (count)
X_raw = np.zeros((n_samples, 3))
X_raw[:, 0] = np.random.uniform(1, 15, size=n_samples)       # Experience
X_raw[:, 1] = np.random.uniform(50, 100, size=n_samples)     # Test Score
X_raw[:, 2] = np.random.randint(0, 5, size=n_samples)        # Certifications

# Target: Hired (0 or 1) - Based on a linear boundary with added noise
noise = np.random.normal(0, 2, size=n_samples)
hired_metric = 0.5 * X_raw[:, 0] + 0.1 * X_raw[:, 1] + 1.2 * X_raw[:, 2] + noise
y = (hired_metric > 12).astype(int)

df = pd.DataFrame(X_raw, columns=['Experience_Years', 'Test_Score', 'Certifications_Count'])
df['Hired'] = y
print(df.head(), "\n")

# 2. Data Splitting & Feature Scaling
X = df.drop(columns=['Hired'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training - Model A: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

# 4. Model Training - Model B: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)

# 5. Performance Evaluation & Matrix Extraction
metrics = {}
for name, preds in [("Random Forest", rf_preds), ("K-Nearest Neighbors", knn_preds)]:
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds)
    }

# Display Comparative Metrics Table
print("--- Step 2: Comparative Performance Matrix ---")
metrics_df = pd.DataFrame(metrics).T
print(metrics_df.round(4))

print("\n--- Model Evaluation Complete. Scripts executed cleanly. ---")
