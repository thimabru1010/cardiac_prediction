import numpy as np
# import torch
# import torch.nn as nn
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classify_scores import classify
from calculate_score import density_factor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
# import train_data_split
# import train_data_loader
import argparse


if __name__=='__main__':
    argparser = argparse.ArgumentParser(description='Classify the calcium scores')
    argparser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'knn', 'svm', 'rf'], help="Model's name")
    argparser.add_argument('--dataset', type=str, default='data/EXAMES/classifier_dataset_radius=120.csv', help='Dataset path to load csv files')
    argparser.add_argument('--pca', action='store_true', help='Apply PCA to the data')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    args = argparser.parse_args()
    
    # Load the data
    # radius = 120
    # df = pd.read_csv(f'data/EXAMES/classifier_dataset_radius={radius}.csv')
    df = pd.read_csv(args.dataset)
    print(df.head(10))
    df['Label'] = df['Escore'].apply(lambda x: classify(x))
    # pacient = 176063
    # print(df[df['Pacient'] == pacient])
    # 1/0
    
    # print(df.groupby('Pacient')['Label'].apply(lambda x: x.value_counts()))
    class_balance = df.groupby('Pacient')['Label'].value_counts().reset_index()['Label'].value_counts()
    print(class_balance)
    # Count the number of samples in each class per pacient
    print(df.groupby('Pacient')['Label'].value_counts().reset_index()['Label'].value_counts(normalize=True))
    # data = np.load('data/EXAMES/train_circle_data.npy')
    print(df.head(10))
    print(df.shape)
    
    # df['density_factor'] = df['Max HU'].apply(lambda x: density_factor(x))
    # df['Agatston Pred'] = df['density_factor'] * df['Area']
    # # Normalize the data
    # # df['Max HU'] = (df['Max HU'] - df['Max HU'].mean()) / df['Max HU'].std()
    # df['Max HU'] = (df['Max HU'] - 130) / 3000
    # df['Centroid X'] = (df['Centroid X']) / 512
    # df['Centroid Y'] = (df['Centroid Y']) / 512
    # df['Area'] = (df['Area'] - df['Area'].mean()) / df['Area'].std()
    # df['Channel'] = (df['Channel'] - df['Channel'].min()) / (df['Channel'].max() - df['Channel'].min())
    # df['Agatston Pred'] = (df['Agatston Pred'] - df['Agatston Pred'].mean()) / df['Agatston Pred'].std()
    # # Convert column Clusters to dummies
    # df = pd.get_dummies(df, columns=['Cluster'])
    
    # print(df.head(10))
    # 1/0
        
    # Get list of non repeated pacients
    # pacients = df['Pacient'].unique()
    # print(pacients)
    # print()
    # Sort by pacients
    df = df.sort_values(by='Pacient')
    pacient_labels = df.groupby('Pacient')['Label'].first().reset_index()
    pacients = pacient_labels['Pacient'].values
    pacient_labels = pacient_labels['Label'].values
    print(pacients)
    print(pacient_labels)
    
    if args.pca:
        pca = PCA(n_components=0.99999)
        # df_pca = pd.DataFrame([], columns=['Pacient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Agatston Pred', 'Channel', 'PCA Col'])
        pca_components = []
        for pacient in pacients:
            # print(pacient)
            X = df[df['Pacient'] == pacient][['Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Agatston Pred', 'Channel']].values
            # print(f"Original Shape: {X.shape}")
            X = X.transpose()
            # print(X.shape)
            pca.fit(X)
            
            # Insert the PCA components into the dataframe
            components = pca.transform(X).transpose()
            # print(components.shape)
            for comp in range(components.shape[0]):
                pca_components.append([pacient] + components[comp, :].tolist() + [comp])
                
        pca_components = np.stack(pca_components, axis=1).transpose()
        print(pca_components.shape)
        df_pca = pd.DataFrame(pca_components, columns=['Pacient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Agatston Pred', 'Channel', 'PCA Comp'])
        print(f"PCA Dataframe shape: {df_pca.shape}")
        df = pd.merge(df[['Pacient', 'Label']].drop_duplicates(), df_pca, on=['Pacient'], how='left')
        print(df.shape)
        print(df.head(10))
                
    # Split the data
    # print(pacients)
    while True:
        pacients_train, pacients_test, y_train, y_test = train_test_split(pacients, pacient_labels, test_size=0.25, stratify=pacient_labels)
        print(f"Train/Test Split: {len(pacients_train)}/{len(pacients_test)}")
        if 176063 in pacients_test:
            break
    # print(len(pacients_train), len(pacients_test))
    
    # df['XY'] = df['Centroid X'] * df['Centroid Y']
    # variables = ['Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Agatston Pred', 'Channel', 'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']
    # variables = ['Latent Factor 1', 'Latent Factor 2', 'Latent Factor 3']
    variables = df.iloc[:, 2:].columns.tolist()
    
    # print(df.groupby('Pacient').size().max())
    features_size = df.groupby('Pacient').size().max() * len(variables)
    print(f"Number of Features: {features_size}")
    
    # Sort the data by pacient
    # df = df.sort_values(by='Pacient')
    
    df_train = df[df['Pacient'].isin(pacients_train)].groupby('Pacient')
    X_train = df_train[variables].apply(lambda x: x.values).values
    # y_train = df_train['Label'].first().values
    # X_train = X_train.values
    print(f"Train Data Shape: {X_train.shape}")
    
    # Pad the data
    X_train_padded = np.zeros((X_train.shape[0], features_size))
    for i, x in enumerate(X_train):
        # print(x.shape)
        x = x.reshape(-1)
        X_train_padded[i, :x.shape[0]] = x
    print(f"Train Padded Data Shape: {X_train_padded.shape}")
    # y_train = df[df['Pacient'].isin(pacients_train)].groupby('Pacient')['Label'].first().values
    # print(len(y_train))
    
    if args.model == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(1000, 100, 50, 10), max_iter=100, random_state=42,\
            early_stopping=True, batch_size=8, verbose=True, validation_fraction=0.2, alpha=0.01)
    elif args.model == 'knn':
        # Define the parameter grid to search over
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],         # Different values for 'k'
            'metric': ['euclidean', 'manhattan']      # Distance metrics
        }
        # Initialize the KNN classifier
        model = KNeighborsClassifier(weights='distance')
        #model = KNeighborsClassifier(n_neighbors=10, n_jobs=10, weights='distance', metric='euclidean')
    elif args.model == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],                 # Regularization parameter
            'kernel': ['linear'],     # Different kernel types
            # 'degree': [2, 3, 4],                     # Degree of the polynomial kernel (only for 'poly' kernel)
            'gamma': ['scale', 'auto']            # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        }
        model = SVC(random_state=42)
    elif args.model == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],           # Number of trees in the forest
            'max_depth': [10, 20, 30],          # Maximum depth of the tree; None means no limit
            'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required at a leaf node
            'max_features': ['sqrt', 'log2'],   # Number of features to consider when looking for the best split
            'bootstrap': [True, False]                # Whether bootstrap samples are used when building trees
        }

        model = RandomForestClassifier(criterion='gini', class_weight='balanced', random_state=42, verbose=1)

    # Train model
    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,                  # Number of folds for cross-validation
        scoring='accuracy',    # Evaluation metric
        n_jobs=-1              # Use all available CPU cores
    )
    
    # Fit grid search on the training data
    print('Starting Grid Search')
    grid_search.fit(X_train_padded, y_train)
    
    # Print the best parameters and best cross-validation score
    # best_params = grid_search.best_params_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    print('Training best model')
    best_model.fit(X_train_padded, y_train)
    
    # Preprocess test data
    df_test = df[df['Pacient'].isin(pacients_test)].groupby('Pacient')
    
    # print(X_test.shape)
    # y_test = df_test['Label'].first().values
    X_test = df_test[variables].apply(lambda x: x.values).values

    # Pad Test Data
    X_test_padded = np.zeros((X_test.shape[0], features_size))
    for i, x in enumerate(X_test):
        # print(x.shape)
        x = x.reshape(-1)
        X_test_padded[i, :x.shape[0]] = x
    
    y_pred = best_model.predict(X_test_padded)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy:.3f}")
    # print(f"Precision: {test_precision:.3f}")
    # print(f"Recall: {test_recall:.3f}")
    print(f"F1 Score: {test_f1:.3f}")
    
    # Save classifier predictions
    # df = pd.read_csv('data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it=5_dilate_k=5.csv')
    results = pd.DataFrame({'Pacient': pacients_test, 'Label': y_test, 'Prediction': y_pred})
    # df = pd.merge(df, results, on='Pacient', how='left')
    dataset_filename = args.dataset.split('/')[-1]
    results.to_csv(f'data/EXAMES/preds_{args.model}_pca={args.pca}_{dataset_filename}', index=True)
    
    # Compare Accuracy to Lesion Model baseline
    df_baseline = pd.read_csv('data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it=5_dilate_k=5.csv')
    df_baseline = df_baseline[df_baseline['Pacient'].isin(pacients_test)]
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
    
    
    
    
    
    