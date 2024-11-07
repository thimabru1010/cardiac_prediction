import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
from numpy import linalg as LA
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import pandas as pd
import seaborn as sns

def bland_altman_plot(data1, data2, * , limit=1.96, title="Bland-Altman Plot"):
    """
    Cria um gráfico de Bland-Altman para avaliar a concordância entre dois conjuntos de dados.

    Parâmetros:
    - data1, data2: As duas listas ou arrays com os valores de medição para comparação.
    - limit: Valor do limite de concordância (default é 1.96, que corresponde ao intervalo de 95% para dados normais).
    - title: Título do gráfico.
    """
    
    # Calcula a média e a diferença entre as duas medições
    data1, data2 = np.asarray(data1), np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)  # Média das diferenças
    sd = np.std(diff)   # Desvio padrão das diferenças

    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.5, marker='o', color='blue', label="Diferença")
    plt.axhline(md, color='red', linestyle='--', label="Média da Diferença")
    plt.axhline(md + limit * sd, color='gray', linestyle='--', label=f"+{limit} SD")
    plt.axhline(md - limit * sd, color='gray', linestyle='--', label=f"-{limit} SD")
    
    # Adiciona títulos e legenda
    plt.title(title)
    plt.xlabel("Média das Medidas")
    plt.ylabel("Diferença entre as Medidas")
    plt.legend()
    plt.show()

def bland_altman_plot_percentiles(data1, data2, title="Bland-Altman Plot with Percentiles"):
    data1, data2 = np.asarray(data1), np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.median(diff)  # Mediana das diferenças
    lower_limit, upper_limit = np.percentile(diff, [2.5, 97.5])  # Limites de concordância com percentis
    print(f"Lower Limit: {lower_limit}, Upper Limit: {upper_limit}")
    # Defining acceptable limits of concordance
    acceptable_limit = 100

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.5, marker='o', color='blue', label="Diferença")
    plt.axhline(md, color='red', linestyle='--', label="Mediana da Diferença")
    plt.axhline(upper_limit, color='gray', linestyle='--', label=f"Limite Superior (97.5%): {lower_limit:.2f}")
    plt.axhline(lower_limit, color='gray', linestyle='--', label=f"Limite Inferior (2.5%): {upper_limit:.2f}")
    
    # Adding acceptable limits of concordance lines
    # plt.axhline(acceptable_limit, color='blue', linestyle='-.', label='Acceptable Limit: +100')
    # plt.axhline(-acceptable_limit, color='blue', linestyle='-.', label='Acceptable Limit: -100')
    
    # Adding vertical lines for calcium score risk thresholds
    plt.axvline(100, color='orange', linestyle='-.', label='Risk Threshold: 100')
    plt.axvline(400, color='purple', linestyle='-.', label='Risk Threshold: 400')

    plt.title(title)
    plt.xlabel("Média das Medidas")
    plt.ylabel("Diferença entre as Medidas")
    plt.legend()
    plt.show()
    fig.savefig('data/EXAMES/bland_altman_plot_percentiles.png', dpi=300)
    plt.close()

def calculate_icc(data1, data2):
    n = len(data1)
    mean_x = np.mean(data1)
    mean_y = np.mean(data2)
    
    ss_total = sum((data1 - mean_x)**2 + (data2 - mean_y)**2)
    ss_within = sum((data1 - data2)**2)
    ms_within = ss_within / n
    ms_between = (ss_total - ss_within) / (n - 1)
    
    return (ms_between - ms_within) / (ms_between + ms_within)

if __name__ == '__main__':
    best_method_path = 'data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it=5_dilate_k=5.csv'
    df_best = pd.read_csv(best_method_path)
    
    # ref_path = 'data/EXAMES/cac_score_data.xlsx'
    # df_ref = pd.read_excel(ref_path)
    # df_ref = pd.merge(df_ref, df_best_method, on='Paciente', how='left')
    # print(df_ref.head())
    
    df_best['Log Escore'] = np.log1p(df_best['Escore'])
    # plot the relative frequency (aka histogram) of Escore values
    plt.figure(figsize=(10, 6))
    sns.histplot(df_best['Log Escore'], kde=True)
    plt.title('Histogram of Escore values')
    plt.show()
    plt.close()
    
    # Calculate ICC
    icc_value = calculate_icc(df_best['Escore'].values, df_best['Lesion Gated'].values)
    print(f"ICC Value: {icc_value}")
    
    bland_altman_plot_percentiles(df_best['Escore'].values, df_best['Lesion Gated'].values,\
        title=f"Bland-Altman Plot with Percentiles (ICC={icc_value:.2f})")
    
    # Calculate the bland-Altmann plot
    # df_best['Difference'] = df_best['Lesion Gated'] - df_best['Escore']
    
    # df_best['Mean'] = (df_best['Lesion Gated'] + df_best['Escore']) / 2
    
    
    
    
        
    
    