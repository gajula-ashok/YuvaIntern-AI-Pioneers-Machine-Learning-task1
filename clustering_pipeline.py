import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Generate Synthetic Unsupervised Clusters
print("--- Step 1: Generating Unlabeled Cluster Datasets ---")
np.random.seed(42)

# Create 3 distinct target groups (e.g., Customer Spending vs Income Profiles)
cluster_0 = np.random.normal(loc=[20, 30], scale=3, size=(50, 2))
cluster_1 = np.random.normal(loc=[60, 70], scale=4, size=(60, 2))
cluster_2 = np.random.normal(loc=[40, 40], scale=3, size=(40, 2))

X_raw = np.vstack([cluster_0, cluster_1, cluster_2])
true_labels = np.concatenate([np.zeros(50), np.ones(60), np.zeros(40) + 2])

df = pd.DataFrame(X_raw, columns=['Annual_Income_K', 'Spending_Score_1_100'])
print(f"Dataset generated cleanly. Total observations: {df.shape[0]} rows.\n")

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Apply K-Means Clustering
# Defining 3 clusters based on our structured domain knowledge
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(X_scaled)
df['Predicted_Cluster'] = predicted_labels

# 4. Unsupervised Model Evaluation
sil_score = silhouette_score(X_scaled, predicted_labels)
print("--- Step 2: Unsupervised Evaluation Metrics ---")
print(f"Calculated Silhouette Score: {sil_score:.4f}")
print("(Note: A score closer to 1 implies highly distinct, well-separated data clusters)\n")

# 5. Mapping and Confusion Matrix Alignment
# Relabeling matching pairs to evaluate accuracy metrics
mapping = {}
for i in range(3):
    labels_in_cluster = true_labels[predicted_labels == i]
    if len(labels_in_cluster) > 0:
        mapping[i] = pd.Series(labels_in_cluster).mode()[0]

mapped_predictions = np.array([mapping[pred] for pred in predicted_labels])

print("--- Step 3: Cluster Evaluation Confusion Matrix ---")
matrix = confusion_matrix(true_labels, mapped_predictions)
print(matrix)

print("\n--- Step 4: Secondary Performance Evaluation ---")
print(classification_report(true_labels, mapped_predictions, target_names=['Cluster 0', 'Cluster 1', 'Cluster 2']))
