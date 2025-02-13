import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import argparse
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

def plot_tpr_tnr_curve(y_true, y_scores, title="TPR vs TNR Curve"):
    """
    Plota a curva TPR (Sensibilidade) versus TNR (Especificidade) variando o threshold.
    
    Parâmetros:
    - y_true: Valores verdadeiros das classes (binárias: 0 ou 1).
    - y_scores: Pontuações de probabilidade do modelo para a classe positiva.
    """
    # Calcula as taxas TPR (Recall) e FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calcula TNR como (1 - FPR)
    tnr = 1 - fpr

    # Plotando TPR versus TNR
    fig = plt.figure(figsize=(8, 6))
    plt.plot(tnr, tpr, label="TPR x TNR", marker=".")
    plt.xlabel("TNR (Especificidade)")
    plt.ylabel("TPR (Sensibilidade)")
    plt.title("Curva TPR vs TNR")
    plt.grid(True)
    plt.legend()
    plt.show()
    fig.savefig(f'data/EXAMES/Experiments_Metrics/{title}.png', dpi=300)
    plt.close()
    
def linear_corr_plot(valores_reais, valores_estimados, title="Linear Correlation Plot", save_path=None):
    # Trace a melhor reta que se ajusta aos dados
    slope, intercept = np.polyfit(valores_estimados, valores_reais, 1)
    
    max_val = max(max(valores_reais), max(valores_estimados))
    ideal_line = np.linspace(0, max_val, 100)
    # Criando o scatter plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(valores_estimados, valores_reais, alpha=0.7, edgecolor='k')
    plt.plot(ideal_line, 
            ideal_line, 
            color='red', linestyle='--', label='Ideal line (y=x)')
    best_fit = slope*valores_estimados + intercept
    plt.plot([min(valores_estimados), max(valores_estimados)], [min(best_fit), max(best_fit)], color='blue', label='Best Fit', linestyle='--')
    
    # Mostre o grafico scatter plot numa janela quadrada, se guiando pelo max dos valores de x e y
    margin = 100
    plt.xlim(-margin, max_val + margin)
    plt.ylim(-margin, max_val + margin)
    
    title = title + f"- Slope={slope:.3f} - Linear Regression Plot"
    plt.title(title)
    plt.xlabel("Estimated Values")
    plt.ylabel("Real Values")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)
    plt.close()


def bland_altman_plot(data1, data2, limit_of_agreement=1.96, title="Bland-Altman Plot", save_path=None):
    """
    Gera um gráfico de Bland-Altman para comparar dois conjuntos de dados.

    Parâmetros:
    - data1, data2: Arrays ou listas com os dois métodos de medição.
    - limit_of_agreement: Multiplicador para calcular os limites de concordância (1.96 para 95% de confiança).
    - title: Título do gráfico.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Calcula as médias e as diferenças
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # Calcula os limites de concordância
    loa_upper = mean_diff + limit_of_agreement * std_diff
    loa_lower = mean_diff - limit_of_agreement * std_diff

    # Criação do gráfico
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.5, label="Diferenças")
    plt.axhline(mean_diff, color="red", linestyle="--", label=f"Média das diferenças ({mean_diff:.2f})")
    plt.axhline(loa_upper, color="blue", linestyle="--", label=f"Limite superior ({loa_upper:.2f})")
    plt.axhline(loa_lower, color="green", linestyle="--", label=f"Limite inferior ({loa_lower:.2f})")
    
    plt.title(title)
    plt.xlabel("Média das medições")
    plt.ylabel("Diferença entre medições")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    # fig.savefig(f'data/EXAMES/Experiments_Metrics/{exam_type}/{avg_str}/{title}.png', dpi=300)
    fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)
    plt.close()

def classify(x, mode=1):
    if mode == 1:
        return classify1(x)
    elif mode == 2:
        return classify2(x)
    
    
def classify1(x):
    if x >= 0 and x < 100: # 2.5
        # return 'sem_risco'
        return 0
    elif x >= 100 and x < 400: # 2.5 - 10
        # return 'risco_intermediario'
        return 1
    elif x >= 400: # >= 10
        # return 'alto_risco'
        return 2
    
def classify2(x):
    if x == 0: # 2.5
        # return 'sem_risco'
        return 0
    elif x > 0 and x <= 400: # 2.5 - 10
        # return 'risco_intermediario'
        return 1
    elif x > 400: # >= 10
        # return 'alto_risco'
        return 2

def get_integers_from_string(s, key):
    integers = s.split(key)[1][:2]
    try:
        integers = int(integers)
    except ValueError:
        integers = int(integers[0])
    return integers
                
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    argparser = argparse.ArgumentParser(description='Classify the calcium scores')
    # argparser.add_argument('--scores_path', type=str, default='data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations.csv', help='CSV filepath with the calcium scores')
    argparser.add_argument('--folder_path', type=str, default='data/EXAMES/Calcium_Scores_Estimations/', help='Folder path to load csv files')
    argparser.add_argument('--fake_gated', action='store_true', help='Whether classify fake_gated scores')
    argparser.add_argument('--avg4', action='store_true', help='Whether to use partes_moles exams averaged by 4 slices (3.2mm)')
    argparser.add_argument('--threshold', '-th', type=str, default=130, help='Calcification Threshold in HU')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    argparser.add_argument('--clssf_mode', '-clssf', type=int, default=1, help='Calcification Threshold in HU')
    args = argparser.parse_args()
    
    avg_str = 'avg=4' if args.avg4 else 'All Slices'
    print(avg_str)
    exam_type = 'Fake_Gated' if args.fake_gated else 'Gated'
    folder_path = os.path.join(args.folder_path, exam_type, avg_str, args.threshold)
    files = os.listdir(folder_path)
    threshold = args.threshold
    # files = ['calcium_score_estimations_dilate_it=5_dilate_k=7.csv']
    max_f1 = 0
    max_acc = 0
    max_f1_error = 0
    max_f1_method = ''
    max_acc_method = ''
    best_filename = ''
    metrics = []
    print(files)
    exp_root_path = f'data/EXAMES/Experiments_Metrics/{exam_type}/{avg_str}/{threshold}/clssf_mode={args.clssf_mode}'
    if not os.path.exists(exp_root_path):
        os.makedirs(exp_root_path)
    for filename in files:
        print(filename)
        # scores_path = args.scores_path
        scores_path = os.path.join(folder_path, filename)
        
        dilate_it = None
        dilate_k = None
        exp_name = "Experiment No Dilation"
        run_name = "No Dilation"
        if 'dilate' in scores_path:
            dilate_it = get_integers_from_string(scores_path, 'dilate_it=')
            dilate_k = get_integers_from_string(scores_path, 'dilate_k=')
            print(dilate_it, dilate_k)
                
            exp_name = f"Experiment dilated it={dilate_it} k={dilate_k}"
            run_name = f"dilated it={dilate_it} k={dilate_k}"
            
        mlflow.set_experiment(exp_name)
        df = pd.read_csv(scores_path)
        df.columns = df.columns.str.strip()
        
        # Change column names
        df.rename(columns={'Pacient': 'Patient', 'ROI Gated': 'ROI', 'Lesion Gated': 'Lesion', 'Heart Mask Gated': 'Heart Mask'}, inplace=True)
        
        print(df['Escore'].head())
        print(df.columns)
        df['Label'] = df['Escore'].apply(lambda x: classify(x, args.clssf_mode))
        df['Lesion Clssf'] = df['Lesion'].apply(lambda x: classify(x, args.clssf_mode))
        df['Heart Clssf'] = df['Heart Mask'].apply(lambda x: classify(x, args.clssf_mode))
        
        # Count the number of samples in each class
        print(df['Label'].value_counts())

        acc_lesion_gated = (df['Label'] == df['Lesion Clssf']).sum() / len(df)
        acc_heart_gated = (df['Label'] == df['Heart Clssf']).sum() / len(df)
        f1_score_lesion_gated = f1_score(df['Label'].values, df['Lesion Clssf'].values, average='weighted')
        avg_error = df['Lesion Error'].mean()
        
        print(f'Accuracy Lesion: {acc_lesion_gated}')
        print(f'F1 Score Lesion: {f1_score_lesion_gated}')
        print(f'Avg Error: {avg_error}')    
        
        metrics.append([run_name, acc_lesion_gated, f1_score_lesion_gated, avg_error])
        #! Confusion Matrices
        # Define class names
        # class_names = ['No Risk', 'Risk 1', 'Risk 2', 'Risk 3', 'Risk 4', 'Risk 5']
        class_names = ['No Risk', 'Risk 1', 'Risk 2']

        # Compute confusion matrix for Lesion Gated
        cm = confusion_matrix(df['Label'].values, df['Lesion Clssf'].values, normalize='true')
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        # Plot the confusion matrix
        figure = plt.figure(figsize=(9,6))
        sns.heatmap(cm_df, annot=True, cmap='Blues')
        plt.title(f'Lesion Confusion Matrix {exam_type} {avg_str} {run_name}')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        # plt.show()
        # cm_folder_path = f'data/EXAMES/confusion_matrices/{exam_type}/{avg_str}/{args.threshold}'
        cm_path = os.path.join(exp_root_path, 'Confusion_Matrices')
        if not os.path.exists(cm_path):
            os.makedirs(cm_path)
        if dilate_it is not None and dilate_k is not None:
            cm_path = os.path.join(cm_path, f'confusion_matrix_dilated_it={dilate_it}_k={dilate_k}.png')
        else:
            cm_path = os.path.join(cm_path, 'confusion_matrix.png')
        figure.savefig(cm_path, dpi=300)
        plt.close()

        # Calculates sensibility and specificity per class
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        
        print(f'Sensitivity: {Sensitivity}')
        print(f'Specificity: {Specificity}')
        
        if max_f1 < f1_score_lesion_gated:
            max_f1 = f1_score_lesion_gated
            max_f1_method = run_name
            max_f1_error = avg_error
            print(f'Max F1 Score: {max_f1}')
            best_filename = filename
            best_sensitivity = Sensitivity
            best_specificity = Specificity
            max_acc = acc_lesion_gated
            print(f'Max F1 Accuracy: {max_acc}')
            print(f'Max F1 Avg Error: {max_f1_error}')
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("exam_type", f"{exam_type}")
            mlflow.set_tag("model", f"{run_name}")
            # mlflow.set_tag("experiment_type", "baseline")
    
            # mlflow.log_metric("ROI ACC", acc_roi_gated)
            mlflow.log_metric("Lesion ACC", acc_lesion_gated)
            # mlflow.log_metric("ROI F1 Score", f1_score_roi_gated)
            mlflow.log_metric("Lesion F1 Score", f1_score_lesion_gated)
            # mlflow.log_artifact(roi_gated_cm_path)
            mlflow.log_artifact(cm_path)

    print()
    print(f'Best method F1: {max_f1_method} - F1 Score: {max_f1} - Accuracy: {max_acc} - Sensitivity: {best_sensitivity} - Specificity: {best_specificity}')
    # print(f"Best method Accuracy: {max_acc_method} - with Accuracy: {max_acc}")
    
    max_acc_method = max_f1_method
    # Load the best method
    filename = best_filename
    print(f"Best filename: {filename}")
    scores_path = os.path.join(folder_path, filename)
    df = pd.read_csv(scores_path)
    df.columns = df.columns.str.strip()
    # Change column names
    df.rename(columns={'Pacient': 'Patient', 'ROI Gated': 'ROI', 'Lesion Gated': 'Lesion', 'Heart Mask Gated': 'Heart Mask'}, inplace=True)
    
    linear_corr_plot(df['Escore'], df['Lesion'], title=f'{exam_type} {avg_str} {max_f1_method} ',\
        save_path=f'data/EXAMES/Experiments_Metrics/{exam_type}/{avg_str}/{threshold}')
    
    bland_altman_plot(df['Escore'], df['Lesion'], title=f'{exam_type} {avg_str} {max_f1_method} - Bland-Altman Plot',\
        save_path=f'data/EXAMES/Experiments_Metrics/{exam_type}/{avg_str}/{threshold}')

    df_best = pd.read_csv(os.path.join(folder_path, best_filename))
    sns.histplot(df_best['Lesion Error'], bins=50)
    plt.title(f'{exam_type} {avg_str} {max_f1_method} - Error Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_root_path, 'error_histogram.png'), dpi=300)
    plt.show()
    plt.close()
    
    #! Plot ACC
    df_metrics = pd.DataFrame(metrics, columns=['Method', 'Accuracy', 'F1 Score', 'Error'])
    # Define a color palette
    palette = sns.color_palette("husl", len(df_metrics))
    fig = plt.figure(figsize=(9,6))
    ax = sns.barplot(x='Method', y='Accuracy', data=df_metrics, palette=palette, hue='Method', legend=False)
    plt.title(f'{exam_type} {avg_str} - Accuracy Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Make the best method bold on the x-axis
    for label in ax.get_xticklabels():
        if label.get_text() == max_acc_method:
            label.set_fontweight('bold')
    plt.savefig(os.path.join(exp_root_path, 'accuracy_metrics.png'), dpi=300)
    plt.show()
    plt.close()
    
    #! Plot F1 Score
    fig = plt.figure(figsize=(9,6))
    ax = sns.barplot(x='Method', y='F1 Score', data=df_metrics, palette=palette, hue='Method', legend=False)
    plt.title(f'{exam_type} {avg_str} - F1 Score Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Make the best method bold on the x-axis
    for label in ax.get_xticklabels():
        if label.get_text() == max_f1_method:
            label.set_fontweight('bold')     
    plt.savefig(os.path.join(exp_root_path, 'f1_metrics.png'), dpi=300)
    plt.show()
    plt.close()
    
    #! Plot Avg Error
    # Plot the average error per method in a barplot
    fig = plt.figure(figsize=(9,6))
    ax = sns.barplot(x='Method', y='Error', data=df_metrics, palette=palette, hue='Method', legend=False)
    plt.title(f'{exam_type} {avg_str} - Average Error Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Make the best method bold on the x-axis
    for label in ax.get_xticklabels():
        if label.get_text() == max_f1_method:
            label.set_fontweight('bold')
    plt.savefig(os.path.join(exp_root_path, 'error_metrics.png'), dpi=300)
    plt.show()
    plt.close()
