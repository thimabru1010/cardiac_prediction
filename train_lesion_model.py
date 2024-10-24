import numpy as np
# import torch
# import torch.nn as nn
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from classify_scores import classify
# import train_data_split
# import train_data_loader
import argparse


if __name__=='__main__':
    argparser = argparse.ArgumentParser(description='Classify the calcium scores')
    argparser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'knn'], help="Model's name")
    argparser.add_argument('--dataset', type=str, default='data/EXAMES/classifier_dataset_radius=120.csv', help='Dataset path to load csv files')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    args = argparser.parse_args()
    
    # Load the data
    # radius = 120
    # df = pd.read_csv(f'data/EXAMES/classifier_dataset_radius={radius}.csv')
    df = pd.read_csv(args.dataset)
    df['Label'] = df['Escore'].apply(lambda x: classify(x))
    # data = np.load('data/EXAMES/train_circle_data.npy')
    print(df.head(10))
    print(df.shape)
    
    # Normalize the data
    df['Max HU'] = (df['Max HU'] - df['Max HU'].mean()) / df['Max HU'].std()
    df['Centroid X'] = (df['Centroid X'] - df['Centroid X'].mean()) / df['Centroid X'].std()
    df['Centroid Y'] = (df['Centroid Y'] - df['Centroid Y'].mean()) / df['Centroid Y'].std()
    df['Area'] = (df['Area'] - df['Area'].mean()) / df['Area'].std()
    df['Channel'] = (df['Channel'] - df['Channel'].mean()) / df['Channel'].std()
    
    # Get list of non repeated pacients
    pacients = df['Pacient'].unique()
    print(pacients)
    # pacients = df['Pacient'].grou
    
    # Split the data
    pacients_train, pacients_test = train_test_split(pacients, test_size=0.2, random_state=42)
    print(len(pacients_train), len(pacients_test))
    
    # X_train = df[['Pacient']]
    # print(X_train.shape)
    features_size = df.groupby('Pacient').size().max() * 4
    print(features_size)
    
    X_train = df[df['Pacient'].isin(pacients_train)]
    
    # print(features_size.max())
    # 1/0
    
    X_train = X_train.groupby('Pacient')[['Max HU', 'Centroid X', 'Centroid Y', 'Area']].apply(lambda x: x.values)
    print(X_train.head())
    
    # X_train = np.stack(X_train.values, axis=0)
    X_train = X_train.values
    print(X_train.shape)
    
    X_train_padded = np.zeros((X_train.shape[0], features_size))
    
    for i, x in enumerate(X_train):
        # print(x.shape)
        x = x.reshape(-1)
        X_train_padded[i, :x.shape[0]] = x
        
    print(X_train_padded.shape)
    y_train = df[df['Pacient'].isin(pacients_train)].groupby('Pacient')['Label'].first().values
    print(len(y_train))
    
    if args.model == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42,\
            early_stopping=True, batch_size=1, verbose=True, validation_fraction=0.2, alpha=0.001)
    elif args.model == 'knn':
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=10)
        
    # Train model
    model.fit(X_train_padded, y_train)
    
    # Preprocess test data
    X_test = df[df['Pacient'].isin(pacients_test)]
    # print(X_test.shape)
    X_test = X_test.groupby('Pacient')[['Max HU', 'Centroid X', 'Centroid Y', 'Area']].apply(lambda x: x.values)
    X_test_padded = np.zeros((X_test.shape[0], features_size))
    for i, x in enumerate(X_test):
        # print(x.shape)
        x = x.reshape(-1)
        X_test_padded[i, :x.shape[0]] = x
    y_test = df[df['Pacient'].isin(pacients_test)].groupby('Pacient')['Label'].first().values
    
    y_pred = model.predict(X_test_padded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Compare Accuracy to Lesion Model baseline
    df_baseline = pd.read_csv('data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it=5_dilate_k=7.csv')
    df_baseline = df_baseline[df_baseline['Pacient'].isin(pacients_test)]
    # print(df_baseline.shape)
    df_baseline['Lesion_preds'] = df_baseline['Lesion Gated'].apply(lambda x: classify(x))
    df_baseline['Label'] = df_baseline['Escore'].apply(lambda x: classify(x))
    accuracy_baseline = accuracy_score(df_baseline['Label'].values, df_baseline['Lesion_preds'].values)
    print(f"Accuracy Baseline (Lesion Gated): {accuracy_baseline:.2f}")
    
    
    
    