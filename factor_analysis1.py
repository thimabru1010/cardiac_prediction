from sklearn.decomposition import FactorAnalysis
import numpy as np
import pandas as pd
from classify_scores import classify
from calculate_score import density_factor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load the newly uploaded CSV file to understand its structure
new_file_path = 'data/EXAMES/classifier_dataset_radius=150_clustered=5.csv'
df = pd.read_csv(new_file_path)

df['density_factor'] = df['Max HU'].apply(lambda x: density_factor(x))
df['Agatston Pred'] = df['density_factor'] * df['Area']
# Normalize the data
# df['Max HU'] = (df['Max HU'] - df['Max HU'].mean()) / df['Max HU'].std()
df['Max HU'] = (df['Max HU'] - 130) / 3000
df['Centroid X'] = (df['Centroid X']) / 512
df['Centroid Y'] = (df['Centroid Y']) / 512
df['Area'] = (df['Area'] - df['Area'].mean()) / df['Area'].std()
df['Channel'] = (df['Channel'] - df['Channel'].min()) / (df['Channel'].max() - df['Channel'].min())
df['Agatston Pred'] = (df['Agatston Pred'] - df['Agatston Pred'].mean()) / df['Agatston Pred'].std()
# Convert column Clusters to dummies
df = pd.get_dummies(df, columns=['Cluster'])
print(df.columns)
    
# Selecting numeric columns for Factor Analysis
numeric_data = df[['Agatston Pred', 'Centroid X', 'Centroid Y', 'Area', 'Channel', 'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']]

# Setting up the Factor Analysis model with a number of components (factors)
n_factors = 3  # Choosing 3 factors for simplicity; this can be adjusted
factor_analysis = FactorAnalysis(n_components=n_factors, random_state=0)

# Fitting the model and transforming the data
factors = factor_analysis.fit_transform(numeric_data)

# Creating a DataFrame for the factor loadings to interpret the factors
factor_loadings = pd.DataFrame(
    factor_analysis.components_.T,
    index=numeric_data.columns,
    columns=[f'Factor {i+1}' for i in range(n_factors)]
)

# Displaying the factor loadings to interpret the factors
# tools.display_dataframe_to_user(name="Factor Loadings", dataframe=factor_loadings)
# Plot factor loadings
factor_loadings.plot(kind='bar', figsize=(10, 6), title='Factor Loadings')
plt.show()
plt.close()


# Assuming df and factors are already defined
# df = pd.DataFrame(...)  # Your DataFrame
# factors = np.array(...)  # Your factors array

# Make a 2D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)

# # Scatter plot
# scatter = ax.scatter(factors[:, 0], factors[:, 1], c=df['Escore'], cmap='viridis', alpha=0.6, edgecolor='k')

# # Add color bar
# cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
# cbar.set_label('Agatston Pred')

# # Set labels
# ax.set_xlabel('Factor 1')
# ax.set_ylabel('Factor 2')

# # Set title
# ax.set_title('2D Scatter Plot of Loadings for 2 Factors')

# # Show plot
# plt.show()
# plt.close()


# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(factors[:, 0], factors[:, 1], factors[:, 2], c=df['Escore'], cmap='viridis', alpha=0.6, edgecolor='k')

# Add color bar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Agatston Escore')

# Set labels
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')

# Set title
ax.set_title('3D Scatter Plot of Loadings for 3 Factors')

# Show plot
plt.show()

