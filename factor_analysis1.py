from sklearn.decomposition import FactorAnalysis
import numpy as np
import pandas as pd
from classify_scores import classify
from calculate_score import density_factor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Re-importing the necessary libraries and re-loading the dataset to resolve the issue
import pandas as pd
from sklearn.decomposition import PCA
# import ace_tools as tools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classify_scores import classify
from calculate_score import density_factor

# Load the previously uploaded file
file_path = 'data/EXAMES/Classifiers_Dataset/classifier_dataset_radius=150_clustered=5.csv'
df = pd.read_csv(file_path)

df['density_factor'] = df['Max HU'].apply(lambda x: density_factor(x))
df['Agatston Pred'] = df['density_factor'] * df['Area']

# Defining bins for the categories (0-50, 50-100, ..., 450-500) for Centroid X and Centroid Y
bins = list(range(0, 550, 50))  # Creating bins up to 500

#! Aggregating Centroid X and Centroid Y into categories
df['Centroid X Cat'] = pd.cut(df['Centroid X'], bins=bins, labels=[f"{i}-{i+50}" for i in bins[:-1]])
df['Centroid Y Cat'] = pd.cut(df['Centroid Y'], bins=bins, labels=[f"{i}-{i+50}" for i in bins[:-1]])
# print(df.head(10))
df_centroids_dummy = pd.get_dummies(df[['Pacient', 'Centroid X Cat', 'Centroid Y Cat']], columns=['Centroid X Cat', 'Centroid Y Cat'])
# print(df_centroids_dummy.head(10))
df_centroids_dummy = df_centroids_dummy.groupby('Pacient').sum()
print(df_centroids_dummy.head(10))

#! Aggregating Channel into categories
print(df['Channel'].describe())
df['Channel'] = (df['Channel'] - df['Channel'].min()) / (df['Channel'].max() - df['Channel'].min())
bins = list(range(0, 50, 5))  # Creating bins up to 500
df['Channel Cat'] = pd.cut(df['Channel'], bins=bins, labels=[f"{i}-{i+5}" for i in bins[:-1]])
df_channel_dummy = pd.get_dummies(df[['Pacient', 'Channel Cat']], columns=['Channel Cat'])
df_channel_dummy = df_channel_dummy.groupby('Pacient').sum()


#! Aggregating Agatston Pred into categories
print(df['Agatston Pred'].describe())
# Defining bins for the Agatston Score Pred with specified intervals
agatston_bins = list(range(0, 101, 10)) + list(range(100, 401, 50)) + list(range(400, 1001, 100)) + list(range(1000, 5001, 1000))
agatston_bins = sorted(list(set(agatston_bins)))
print(agatston_bins)
# Applying the binning to Agatston Score Pred
df['Agatston Pred'] = pd.cut(
    df['Escore'], 
    bins=agatston_bins, 
    labels=[f"{agatston_bins[i]}-{agatston_bins[i+1]}" for i in range(len(agatston_bins)-1)]
)
df_agatston_dummy = pd.get_dummies(df[['Pacient', 'Agatston Pred']], columns=['Agatston Pred'])
# Grouping by patient and summing to get frequency of each score range per patient
df_agatston_dummy = df_agatston_dummy.groupby('Pacient').sum()
print(df_agatston_dummy.head(10))

#! Aggregating the Area into categories
# Centralizando os dados
print(df['Area'].describe())
mean_area = df['Area'].mean()
centralized_area = df['Area'] - mean_area
std_dev = centralized_area.std()
print(std_dev)
# Definindo limites das categorias baseadas na variância
min_value = centralized_area.min()
max_value = centralized_area.max()

bins = []
current = min_value
while current < max_value:
    bins.append(current)
    current += std_dev
bins.append(max_value)

print(len(bins))

# Aplicando a categorização ao dado centralizado
df['Centered Area Cat'] = pd.cut(centralized_area, bins=bins, include_lowest=True)
df_area_dummy = pd.get_dummies(df[['Pacient', 'Centered Area Cat']], columns=['Centered Area Cat'])
df_area_dummy = df_area_dummy.groupby('Pacient').sum()

#! Aggregating the Max HU into categories
print(df['Max HU'].describe())
mean_max_hu = df['Max HU'].mean()
centralized_max_hu = df['Max HU'] - mean_max_hu
std_dev = centralized_max_hu.std()
print(std_dev)
min_value = centralized_max_hu.min()
max_value = centralized_max_hu.max()

bins = []
current = min_value
while current < max_value:
    bins.append(current)
    current += std_dev
bins.append(max_value)
print(len(bins))

df['Centered Max HU Cat'] = pd.cut(centralized_max_hu, bins=bins, include_lowest=True)
df_max_hu_dummy = pd.get_dummies(df[['Pacient', 'Centered Max HU Cat']], columns=['Centered Max HU Cat'])
df_max_hu_dummy = df_max_hu_dummy.groupby('Pacient').sum()

# Merge all the dataframes above
df_data = pd.merge(df_centroids_dummy, df_channel_dummy, on='Pacient')
df_data = pd.merge(df_data, df_area_dummy, on='Pacient')
df_data = pd.merge(df_data, df_max_hu_dummy, on='Pacient')
df_data = pd.merge(df_data, df_agatston_dummy, on='Pacient')
print(df_data.head(10))
print(df_data.shape)

# Drop columns that contain only zeros
df_data = df_data.loc[:, (df_data != 0).any()]

print(df_data.head(10))
print(df_data.shape)

# Normalize data by the maximum value
df_data = df_data / df_data.max()
print(df_data.head(10))

# Selecting numeric columns for Factor Analysis
# numeric_data = df[['Agatston Pred', 'Centroid X', 'Centroid Y', 'Area', 'Channel', 'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']]
numeric_data = df_data.iloc[:, 1:]

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
df_data = pd.merge(df_data, df[['Pacient', 'Escore']], on='Pacient').drop_duplicates()

scatter = ax.scatter(factors[:, 0], factors[:, 1], factors[:, 2], c=df_data['Escore'], cmap='viridis', alpha=0.6, edgecolor='k')

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

