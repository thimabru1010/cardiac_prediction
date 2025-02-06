import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
csv_path = 'data/EXAMES/classifier_dataset_radius=150.csv'
data = pd.read_csv(csv_path)

data['Centroid X'] = (data['Centroid X']) / 512
data['Centroid Y'] = (data['Centroid Y']) / 512

channel_min = data['Channel'].min()
channel_max = data['Channel'].max()
data['Channel'] = (data['Channel'] - channel_min) / (channel_max - channel_min)

# Extract relevant features
clustering_data = data[['Centroid X', 'Centroid Y', 'Channel']]

inertia_values = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k).fit(clustering_data)
    inertia_values.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(clustering_data, kmeans.labels_)
    ch_score = calinski_harabasz_score(clustering_data, kmeans.labels_)
    db_score = davies_bouldin_score(clustering_data, kmeans.labels_)
    print()
    print(f"Number of Clusters: {k}, Inertia: {kmeans.inertia_}, Silhouette Score: {silhouette_avg}")
    print(f"Calinski-Harabasz Score: {ch_score}, Davies-Bouldin Score: {db_score}")

fig = plt.figure(figsize=(10, 8))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
plt.close()

# Apply KMeans clustering
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters)
data['Cluster'] = kmeans.fit_predict(clustering_data)

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Centroid X'], data['Centroid Y'], data['Channel'],\
                     c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolor='k')

# Labels and title
ax.set_xlabel('Centroid X')
ax.set_ylabel('Centroid Y')
ax.set_zlabel('Channel')
ax.set_title('3D Clustering of Centroids by X, Y Coordinates and Channel')

# Add color bar
fig.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show()
plt.close()

# basename = csv_path.split('/')[-1]
# Unnormalize and save clustered data
data['Centroid X'] = data['Centroid X'] * 512
data['Centroid Y'] = data['Centroid Y'] * 512
# data['Channel'] = data['Channel'] * (data['Channel'].max() - data['Channel'].min()) + data['Channel'].min()
data['Channel'] = data['Channel'] * (channel_max - channel_min) + channel_min
data.to_csv(csv_path[:-4] + f"_clustered={n_clusters}.csv", index=False)
