import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import docx

def read_docx_table(file_path):
    doc = docx.Document(file_path)
    table = doc.tables[0]
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    data = {header: [] for header in headers}
    for row in table.rows[1:]:
        for i, cell in enumerate(row.cells):
            data[headers[i]].append(cell.text.strip())
    return pd.DataFrame(data)

# Load and preprocess data
file_path = 'C:\\Users\\Hvl\\Desktop\\dataset.DOCX'
df = read_docx_table(file_path)
numeric_cols = ['A', 'B', 'C', 'D']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
df['Durum'] = df['Durum'].str.lower().str.replace('hastadegil', 'sağlıklı').str.replace('hasta', 'hastalıklı')

# Standardize data
scaler = StandardScaler()
X = df[numeric_cols]
X_scaled = scaler.fit_transform(X)

# K-Medoids clustering
k = 2
initial_medoids = np.random.choice(range(len(X_scaled)), size=k, replace=False)
kmedoids_instance = kmedoids(X_scaled.tolist(), initial_medoids.tolist())
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Create cluster labels
labels = np.zeros(len(X_scaled), dtype=int)
for cluster_idx, cluster in enumerate(clusters):
    labels[cluster] = cluster_idx
df['Cluster'] = labels

# Align clusters with true labels
true_labels = df['Durum'].map({'sağlıklı': 0, 'hastalıklı': 1}).values

def align_clusters(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    new_labels = np.zeros_like(cluster_labels)
    for i, j in zip(row_ind, col_ind):
        new_labels[cluster_labels == j] = i
    return new_labels

aligned_clusters = align_clusters(true_labels, df['Cluster'])
df['Aligned_Cluster'] = aligned_clusters

# Convert numpy array to pandas Series for mapping
aligned_clusters_series = pd.Series(aligned_clusters)

# Visualization 1: Heatmaps
plt.figure(figsize=(18, 12))

# 1.1 Correlation Heatmap
plt.subplot(2, 3, 1)
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Özellikler Arası Korelasyon')

# 1.2 Cluster Feature Heatmap
plt.subplot(2, 3, 2)
cluster_means = df.groupby('Cluster')[numeric_cols].mean()
sns.heatmap(cluster_means.T, annot=True, cmap='viridis', fmt=".2f")
plt.title('Küme Özellik Ortalamaları')

# 1.3 Data Distribution Heatmap
plt.subplot(2, 3, 3)
df_sorted = df.sort_values('Cluster')
sns.heatmap(df_sorted[numeric_cols], cmap='plasma', yticklabels=False)
plt.title('Veri Dağılımı')

# Visualization 2: True vs Predicted
# 2.1 Confusion Matrix
plt.subplot(2, 3, 4)
cm = confusion_matrix(true_labels, aligned_clusters)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sağlıklı (Tahmin)', 'Hastalıklı (Tahmin)'],
            yticklabels=['Sağlıklı (Gerçek)', 'Hastalıklı (Gerçek)'])
plt.title('Karışıklık Matrisi')

# 2.2 Class Distribution
plt.subplot(2, 3, 5)
comparison_df = df.copy()
comparison_df['Predicted'] = aligned_clusters_series.map({0: 'Sağlıklı', 1: 'Hastalıklı'})  # Fixed this line
sns.countplot(data=comparison_df, x='Durum', hue='Predicted', palette='Set2')
plt.title('Gerçek vs Tahmini Dağılım')

plt.tight_layout()
plt.show()

# Visualization 3: Additional Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 3.1 Feature Scatter
sns.scatterplot(ax=axes[0,0], data=df, x='A', y='B', hue='Cluster', style='Durum', palette='viridis')
axes[0,0].set_title('A vs B Dağılımı')

# 3.2 Feature Scatter with Medoids
sns.scatterplot(ax=axes[0,1], data=df, x='C', y='D', hue='Cluster', style='Durum', palette='viridis')
medoids_orig = scaler.inverse_transform(np.array([X_scaled[i] for i in medoids]))
axes[0,1].scatter(medoids_orig[:,2], medoids_orig[:,3], c='red', marker='X', s=200, label='Medoidler')
axes[0,1].set_title('C vs D Dağılımı')
axes[0,1].legend()

# 3.3 Performance Metrics
metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
scores = [silhouette_score(X_scaled, df['Cluster']),
          davies_bouldin_score(X_scaled, df['Cluster']),
          calinski_harabasz_score(X_scaled, df['Cluster'])]
sns.barplot(ax=axes[1,0], x=metrics, y=scores, palette='rocket')
axes[1,0].set_title('Kümeleme Metrikleri')

# 3.4 Dendrogram
Z = linkage(squareform(pdist(X_scaled)), 'ward')
dendrogram(ax=axes[1,1], Z=Z, truncate_mode='lastp', p=12)
axes[1,1].set_title('Hiyerarşik Kümeleme')

plt.tight_layout()
plt.show()

# Print results
print("\nKümeleme Sonuçları:")
print(pd.crosstab(df['Durum'],
                 aligned_clusters_series.map({0:'Sağlıklı',1:'Hastalıklı'}),
                 margins=True))
print(f"\nSilhouette: {scores[0]:.3f}")
print(f"Davies-Bouldin: {scores[1]:.3f}")
print(f"Calinski-Harabasz: {scores[2]:.3f}")