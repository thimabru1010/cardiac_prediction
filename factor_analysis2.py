# Re-importing the necessary libraries and re-loading the dataset to resolve the issue
import pandas as pd
from sklearn.decomposition import PCA
# import ace_tools as tools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classify_scores import classify

# Load the previously uploaded file
file_path = 'data/EXAMES/classifier_dataset_radius=150.csv'
new_data = pd.read_csv(file_path)

# Extract unique patient data by taking the mean for each patient
patient_data = new_data.groupby('Pacient').mean()  # Average across each patient
patient_data = patient_data[['Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Channel', 'Escore']]

# Setting up PCA to reduce dimensions to represent each patient with latent factors
n_components = 3  # Reducing to 3 components (latent factors)
pca = PCA(n_components=n_components, random_state=0)
patient_factors = pca.fit_transform(patient_data)

# Creating a DataFrame to show the reduced dimensional representation of each patient
patient_factors_df = pd.DataFrame(patient_factors, index=patient_data.index, columns=[f'Latent Factor {i+1}' for i in range(n_components)])

pacient_labels = new_data[['Pacient', 'Escore']].drop_duplicates()
pacient_factors_df = pd.merge(pacient_labels                                    , patient_factors_df, on='Pacient')
print(pacient_factors_df.head())

dataset_basename = file_path.split('/')[-1].split('.')[0]
pacient_factors_df.to_csv(f'data/EXAMES/pacient_factors_{dataset_basename}.csv', index=False)
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
                c=patient_data['Escore'], 
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