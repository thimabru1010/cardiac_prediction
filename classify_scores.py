import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import argparse
import mlflow
import os

def classify(x):
    if x >= 0 and x < 100:
        # return 'sem_risco'
        return 0
    elif x >= 100 and x < 400:
        # return 'risco_intermediario'
        return 1
    elif x >= 400:
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
    argparser = argparse.ArgumentParser(description='Classify the calcium scores')
    # argparser.add_argument('--scores_path', type=str, default='data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations.csv', help='CSV filepath with the calcium scores')
    argparser.add_argument('--folder_path', type=str, default='data/EXAMES/Calcium_Scores_Estimations/', help='Folder path to load csv files')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    args = argparser.parse_args()
    
    files = os.listdir(args.folder_path)
    files = ['calcium_score_estimations_dilate_it=5_dilate_k=7.csv']
    for filename in files:
        print(filename)
        # scores_path = args.scores_path
        scores_path = os.path.join(args.folder_path, filename)
        
        dilate_it = None
        dilate_k = None
        exp_name = "Experiment No Dilation"
        run_name = "No Dilation"
        if 'dilate' in scores_path:
            # dilate_it = scores_path.split('dilate_it=')[1][:2]
            # try:
            #     dilate_it = int(dilate_it)
            # except ValueError:
            #     dilate_it = int(dilate_it[0])
            dilate_it = get_integers_from_string(scores_path, 'dilate_it=')
            dilate_k = get_integers_from_string(scores_path, 'dilate_k=')
            print(dilate_it, dilate_k)
                
                
            # dilate_k = scores_path.split('dilate_k=')[1][:2]
            # numbers = re.findall(r'\d{1,2}', scores_path)
            exp_name = f"Experiment dilated it={dilate_it} k={dilate_k}"
            run_name = f"dilated it={dilate_it} k={dilate_k}"
            
        mlflow.set_experiment(exp_name)
        df = pd.read_csv(scores_path)
        df.columns = df.columns.str.strip()
        
        print(df['Escore'].head())
        df['Label'] = df['Escore'].apply(lambda x: classify(x))
        df['ROI Gated Clssf'] = df['ROI Gated'].apply(lambda x: classify(x))
        df['Lesion Gated Clssf'] = df['Lesion Gated'].apply(lambda x: classify(x))
        df['Circle Gated Clssf'] = df['Circle Mask Gated'].apply(lambda x: classify(x))
        # df['Estimated Fake Gated Clssf'] = df['Estimated Score Fake Gated'].apply(lambda x: classify(x))
        # df['Lesion Fake Gated Clssf'] = df['Lesion Fake Gated'].apply(lambda x: classify(x))
        
        # Count the number of samples in each class
        print(df['Label'].value_counts())
        # 1/0

        acc_roi_gated = (df['Label'] == df['ROI Gated Clssf']).sum() / len(df) *100
        acc_lesion_gated = (df['Label'] == df['Lesion Gated Clssf']).sum() / len(df) *100
        acc_circle_gated = (df['Label'] == df['Circle Gated Clssf']).sum() / len(df) *100
        # accuracy_efg = (df['Label'] == df['Estimated Fake Gated Clssf']).sum() / len(df) *100
        # accuracy_dfg = (df['Label'] == df['Direct Fake Gated Clssf']).sum() / len(df) *100

        print(f'Accuracy ROI Gated: {acc_roi_gated}%')
        # print(f'Accuracy Estimated Fake Gated: {accuracy_efg}%')
        # print(f'Accuracy Direct Fake Gated: {accuracy_dfg}%')
        print(f'Accuracy Lesion Gated: {acc_lesion_gated}%')
        print(f'Accuracy Circle Mask Gated: {acc_circle_gated}%')

        f1_score_roi_gated = f1_score(df['Label'].values, df['ROI Gated Clssf'].values, average='weighted')
        f1_score_lesion_gated = f1_score(df['Label'].values, df['Lesion Gated Clssf'].values, average='weighted')
        
        #! Confusion Matrices
        # Define class names
        class_names = ['sem_risco', 'risco_intermediario', 'alto_risco']

        # Compute confusion matrix for ROI Gated
        cm = confusion_matrix(df['Label'].values, df['ROI Gated Clssf'].values, normalize='true')
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        # Plot the confusion matrix
        figure = plt.figure(figsize=(9,6))
        sns.heatmap(cm_df, annot=True, cmap='Blues')
        plt.title('ROI Gated Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        # plt.show()
        if dilate_it is not None and dilate_k is not None:
            roi_gated_cm_path = f'data/EXAMES/confusion_matrices/confusion_matrix_ROI_dilated_it={dilate_it}_k={dilate_k}.png'
        else:
            roi_gated_cm_path = 'data/EXAMES/confusion_matrices/confusion_matrix_ROI.png'
        figure.savefig(roi_gated_cm_path, dpi=300)
        plt.close()

        # Compute confusion matrix for Lesion Gated
        cm = confusion_matrix(df['Label'].values, df['Lesion Gated Clssf'].values, normalize='true')
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        # Plot the confusion matrix
        figure = plt.figure(figsize=(9,6))
        sns.heatmap(cm_df, annot=True, cmap='Blues')
        plt.title('Lesion Gated Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        # plt.show()
        if dilate_it is not None and dilate_k is not None:
            lesion_gated_cm_path = f'data/EXAMES/confusion_matrices/confusion_matrix_Lesion_dilated_it={dilate_it}_k={dilate_k}.png'
        else:
            lesion_gated_cm_path = 'data/EXAMES/confusion_matrices/confusion_matrix_Lesion.png'
        figure.savefig(lesion_gated_cm_path, dpi=300)
        plt.close()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("ROI Gated ACC", acc_roi_gated)
            mlflow.log_metric("Lesion Gated ACC", acc_lesion_gated)
            mlflow.log_metric("ROI Gated F1 Score", f1_score_roi_gated)
            mlflow.log_metric("Lesion Gated F1 Score", f1_score_lesion_gated)
            mlflow.log_artifact(roi_gated_cm_path)
            mlflow.log_artifact(lesion_gated_cm_path)
            
        print('ROI Gated Clssf')
        # Alternatively, print classification report
        report = classification_report(df['Label'].values, df['ROI Gated Clssf'].values, target_names=class_names)
        print('Classification Report:\n', report)

        print('Lesion Gated Clssf')
        # Alternatively, print classification report
        report = classification_report(df['Label'].values, df['Lesion Gated Clssf'].values, target_names=class_names)
        print('Classification Report:\n', report)

    # print('Circle Gated Clssf')
    # # Alternatively, print classification report
    # report = classification_report(df['Label'].values, df['Circle Gated Clssf'].values, target_names=class_names)
    # print('Classification Report:\n', report)
