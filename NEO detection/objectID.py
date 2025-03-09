import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA

data= pd.read_csv("neo_data.csv")
print("Data Head",data.head())
print("Data Information", data.info())
print("Data Columns:", data.columns)

data= data.drop(['Min Diameter (feet)','Max Diameter (feet)', 'Min Diameter (m)','Max Diameter (m)','Min Diameter (miles)','Max Diameter (miles)', 'Relative Velocity (miles/h)','Miss Distance (astronomical)','Miss Distance (lunar)'], axis=1)
data= data.drop(['ID', 'Neo Reference ID', 'Name', 'Limited Name', 'NASA JPL URL','Close Approach Date','Close Approach Date (Full)','Epoch Date Close Approach'], axis=1)
data['Is Potentially Hazardous'] = data['Is Potentially Hazardous'].astype(int)
data['Orbiting Body'] = data['Orbiting Body'].astype('category').cat.codes

corr_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

columns_to_normalize= ['Min Diameter (km)','Max Diameter (km)','Absolute Magnitude (H)','Relative Velocity (km/s)','Relative Velocity (km/h)','Miss Distance (km)','Miss Distance (miles)' ]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[columns_to_normalize])
scaled_data = pd.DataFrame(scaled_data, columns=columns_to_normalize)
print(scaled_data.head())

pca = PCA(n_components=3)  # Reducing to 3 dimensions
pca_result = pca.fit_transform(scaled_data)

# Extract the two principal components
pca_1 = pca_result[:, 0]
pca_2 = pca_result[:, 1]
pca_3 = pca_result[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_1, pca_2, pca_3, alpha=0.6, edgecolor='k', s=50, cmap='viridis')
plt.title("PCA Visualization (3D) of Scaled Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.grid(True)
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by component: {explained_variance}")

# Initialize an empty list to store the sum of squared distances (inertia)
inertia = []

# Test k values from 1 to 10 (or higher if needed)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)  # k=3
kmeans.fit(scaled_data)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_
# Add the cluster labels to the original dataset for reference
import pandas as pd
data['Cluster'] = cluster_labels  # Assuming your original dataset is in a DataFrame called 'neo_data'
# Display a few rows to check
print(data.head())

cluster_analysis = data.groupby('Cluster').mean()
print(cluster_analysis)

# Example: Assign labels based on cluster analysis
cluster_labels_mapped = {
    0: 'Other',
    1: 'Comet',
    2: 'Asteroid'  # Update based on your analysis
}
data['Object Type'] = data['Cluster'].map(cluster_labels_mapped)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

data['pca_1'] = pca_1
data['pca_2'] = pca_2
data['pca_3'] = pca_3
# Use Object Type for coloring
for obj_type, color in zip(['Asteroid', 'Comet', 'Other'], ['blue', 'green', 'red']):
    subset = data[data['Object Type'] == obj_type]
    ax.scatter(subset['pca_1'], subset['pca_2'], subset['pca_3'], 
               label=obj_type, alpha=0.6, edgecolor='k', s=50, c=color)

ax.set_title("3D Visualization of NEO Clusters")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.legend()
plt.show()
data.to_csv('labeled_neo_data.csv', index=False)

silhouette_avg = silhouette_score(scaled_data, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")
ch_score = calinski_harabasz_score(scaled_data, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_score:.2f}")

# Combine preprocessing steps (scaler, PCA) and KMeans into a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Feature scaling
    ('pca', PCA(n_components=3)),         # Dimensionality reduction
    ('kmeans', KMeans(n_clusters=3, random_state=42))  # Clustering
])

# Fit the pipeline on your data
pipeline.fit(scaled_data[['Absolute Magnitude (H)','Relative Velocity (km/h)','Min Diameter (km)']])

# Save the pipeline
dump(pipeline, 'object_model_pipeline.joblib')
print("Pipeline saved as 'object_model_pipeline.joblib'")
def map_cluster_to_label(cluster_id):
    cluster_labels_mapped = {
        0: 'Other',
        1: 'Comet',
        2: 'Asteroid'  # Modify based on your analysis
    }
    return cluster_labels_mapped.get(cluster_id, 'Unknown')