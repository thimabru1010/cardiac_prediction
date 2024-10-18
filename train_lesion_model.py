import numpy as np
# import torch
# import torch.nn as nn
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from classify_scores import classify
# import train_data_split
# import train_data_loader


if __name__=='__main__':
    # Load the data
    radius = 120
    df = pd.read_csv(f'data/EXAMES/classifier_dataset_radius={radius}.csv')
    df['Label'] = df['Escore'].apply(lambda x: classify(x))
    # data = np.load('data/EXAMES/train_circle_data.npy')
    print(df.head(10))
    print(df.shape)
    
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
    
    # Train MLP scikitlearn
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=100, random_state=42,\
        early_stopping=True, batch_size=1, verbose=True, validation_fraction=0.2)
    mlp.fit(X_train_padded, y_train)
    
    X_test = df[df['Pacient'].isin(pacients_test)]
    X_test = X_test.groupby('Pacient')[['Max HU', 'Centroid X', 'Centroid Y', 'Area']].apply(lambda x: x.values)
    X_test_padded = np.zeros((X_test.shape[0], features_size))
    for i, x in enumerate(X_test):
        # print(x.shape)
        x = x.reshape(-1)
        X_test_padded[i, :x.shape[0]] = x
    y_test = df[df['Pacient'].isin(pacients_test)].groupby('Pacient')['Label'].first().values
    
    y_pred = mlp.predict(X_test_padded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    
    