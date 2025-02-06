# Re-importing the necessary libraries and re-loading the dataset to resolve the issue
import pandas as pd
from sklearn.decomposition import PCA
# import ace_tools as tools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classify_scores import classify
from calculate_score import density_factor

# Load the previously uploaded file
file_path = 'data/EXAMES/Classifiers_Dataset/classifier_dataset_radius=10000.csv'
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

# Drop columns that contain only zeros
df_data = df_data.loc[:, (df_data != 0).any()]

print(df_data.head(10))
print(df_data.shape)

# Normalize data by the maximum value
df_data = df_data / df_data.max()
print(df_data.head(10))

# df_data = pd.merge(df_centroids_dummy, df_channel_dummy, df_agatston_dummy, on='Pacient')

# Normalize the data
# df['Max HU'] = (df['Max HU'] - df['Max HU'].mean()) / df['Max HU'].std()
# df['Max HU'] = (df['Max HU'] - 130) / 3000
# df['Centroid X'] = (df['Centroid X']) / 512
# df['Centroid Y'] = (df['Centroid Y']) / 512
# df['Area'] = (df['Area'] - df['Area'].mean()) / df['Area'].std()
# df['Channel'] = (df['Channel'] - df['Channel'].min()) / (df['Channel'].max() - df['Channel'].min())
# df['Agatston Pred'] = (df['Agatston Pred'] - df['Agatston Pred'].mean()) / df['Agatston Pred'].std()
# Convert column Clusters to dummies
# df = pd.get_dummies(df, columns=['Cluster'])
# print(df.columns)

# Extract unique patient data by taking the mean for each patient
# patient_data = df.groupby('Pacient').mean()  # Average across each patient
patient_data = df_data.copy()
# patient_data = patient_data[['Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Channel', 'Agatston Pred']]

# Setting up PCA to reduce dimensions to represent each patient with latent factors
n_components = 0.95  # Reducing to 3 components (latent factors)
pca = PCA(n_components=n_components, random_state=0)
# pca = PCA(n_compo)
patient_factors = pca.fit_transform(patient_data.iloc[:, 1:])  # Ignoring the patient ID column
# Get the variance of pca
print(f"Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")
print(pca.explained_variance_ratio_)

# Creating a DataFrame to show the reduced dimensional representation of each patient
# patient_factors_df = pd.DataFrame(patient_factors, index=patient_data.index, columns=[f'Latent Factor {i+1}' for i in range(n_components)])
patient_factors_df = pd.DataFrame(patient_factors, index=patient_data.index)
print(patient_factors_df.shape)

pacient_labels = df[['Pacient', 'Escore']].drop_duplicates()
pacient_factors_df = pd.merge(pacient_labels, patient_factors_df, on='Pacient')
print(pacient_factors_df.head())

dataset_basename = file_path.split('/')[-1].split('.')[0]

# pacient_factors_df = pd.merge(pacient_factors_df, df[['Pacient', 'Escore']], on='Pacient')
pacient_factors_df.to_csv(f'data/EXAMES/pacient_factors_{dataset_basename}.csv', index=False)
1/0
# Displaying the reduced representation to interpret the latent factors for each patient
# tools.display_dataframe_to_user(name="Latent Factors for Patients", dataframe=patient_factors_df)

# Adding color to represent the Agatston scores (Escore) in the visualizations

# 2D Scatter Plots with color representing the Agatston Score
# plt.figure(figsize=(12, 8))

# # Plotting Factor 1 vs Factor 2 with color
# plt.subplot(1, 3, 1)
# sc = plt.scatter(patient_factors_df['Latent Factor 1'], 
#                  patient_factors_df['Latent Factor 2'], 
#                  c=patient_data['Escore'], 
#                  cmap='viridis', alpha=0.7)
# plt.colorbar(sc, label='Agatston Score (Escore)')
# plt.xlabel('Latent Factor 1')
# plt.ylabel('Latent Factor 2')
# plt.title('Latent Factor 1 vs Factor 2')

# # Plotting Factor 1 vs Factor 3 with color
# plt.subplot(1, 3, 2)
# sc = plt.scatter(patient_factors_df['Latent Factor 1'], 
#                  patient_factors_df['Latent Factor 3'], 
#                  c=patient_data['Escore'], 
#                  cmap='viridis', alpha=0.7)
# plt.colorbar(sc, label='Agatston Score (Escore)')
# plt.xlabel('Latent Factor 1')
# plt.ylabel('Latent Factor 3')
# plt.title('Latent Factor 1 vs Factor 3')

# # Plotting Factor 2 vs Factor 3 with color
# plt.subplot(1, 3, 3)
# sc = plt.scatter(patient_factors_df['Latent Factor 2'], 
#                  patient_factors_df['Latent Factor 3'], 
#                  c=patient_data['Escore'], 
#                  cmap='viridis', alpha=0.7)
# plt.colorbar(sc, label='Agatston Score (Escore)')
# plt.xlabel('Latent Factor 2')
# plt.ylabel('Latent Factor 3')
# plt.title('Latent Factor 2 vs Factor 3')

# plt.tight_layout()
# plt.show()

# Creating a 3D scatter plot with Agatston scores as color
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the three latent factors with Agatston score as color
sc = ax.scatter(patient_factors_df['Latent Factor 1'], 
                patient_factors_df['Latent Factor 2'], 
                patient_factors_df['Latent Factor 3'], 
                c=pacient_factors_df['Escore'], 
                cmap='viridis', alpha=0.7)

# Setting axis labels
ax.set_xlabel('Latent Factor 1')
ax.set_ylabel('Latent Factor 2')
ax.set_zlabel('Latent Factor 3')
ax.set_title('3D Visualization of Latent Factors with Agatston Score (Escore)')

# Adding a color bar to represent the Agatston score
cb = fig.colorbar(sc, ax=ax, label='Agatston Score (Escore)')
plt.show()
plt.close()

def prediction(x):
    if x[0] > 0 and x[0] <= 600:
        return 1
    elif x[0] >= 600:
        return 2
    else:
        return 0
    
pacient_factors_df['Preds'] = pacient_factors_df[['Latent Factor 1', 'Latent Factor 2', 'Latent Factor 3']].apply(lambda x: prediction(x.values), axis=1)

y_pred = pacient_factors_df['Preds'].values
pacient_factors_df['Label'] = pacient_factors_df['Escore'].apply(lambda x: classify(x))
y_test = pacient_factors_df['Label'].values

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='macro')
test_recall = recall_score(y_test, y_pred, average='macro')
test_f1 = f1_score(y_test, y_pred, average='macro')

print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.3f}")
# print(f"Precision: {test_precision:.3f}")
# print(f"Recall: {test_recall:.3f}")
print(f"F1 Score: {test_f1:.3f}")

# Compare Accuracy to Lesion Model baseline
df_baseline = pd.read_csv('data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it=5_dilate_k=5.csv')
# df_baseline = df_baseline[df_baseline['Pacient']]
# print(df_baseline.shape)
# X_test = X_test.groupby('Pacient')[['Max HU', 'Centroid X', 'Centroid Y', 'Area']].apply(lambda x: x.values)
df_baseline['Lesion_preds'] = df_baseline['Lesion Gated'].apply(lambda x: classify(x))
df_baseline['Label'] = df_baseline['Escore'].apply(lambda x: classify(x))
accuracy_baseline = (df_baseline['Label'] == df_baseline['Lesion_preds']).sum() / len(df_baseline) *100
baseline_accuracy = accuracy_score(df_baseline['Label'].values, df_baseline['Lesion_preds'].values)
baseline_precision = precision_score(df_baseline['Label'].values, df_baseline['Lesion_preds'].values, average='macro')
baseline_recall = recall_score(df_baseline['Label'].values, df_baseline['Lesion_preds'].values, average='macro')
baseline_f1 = f1_score(df_baseline['Label'].values, df_baseline['Lesion_preds'].values, average='macro')

print('\nBaseline Evaluation:')
print(accuracy_baseline)
print(f"Accuracy: {baseline_accuracy:.3f}")
# print(f"Precision: {baseline_precision:.3f}")
# print(f"Recall: {baseline_recall:.3f}")
print(f"F1 Score: {baseline_f1:.3f}")
# print(f"Accuracy Baseline (Lesion Gated): {accuracy_baseline:.2f}")