import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

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
    
scores_path = 'data/EXAMES/calcium_score_estimations.csv'

df = pd.read_csv(scores_path)

df['Label'] = df['Escore'].apply(lambda x: classify(x))
df['ROI Gated Clssf'] = df['ROI Gated'].apply(lambda x: classify(x))
df['Lesion Gated Clssf'] = df['Lesion Gated'].apply(lambda x: classify(x))
df['Circle Gated Clssf'] = df['Circle Mask Gated'].apply(lambda x: classify(x))
# df['Estimated Fake Gated Clssf'] = df['Estimated Score Fake Gated'].apply(lambda x: classify(x))
# df['Lesion Fake Gated Clssf'] = df['Lesion Fake Gated'].apply(lambda x: classify(x))

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

# Define class names
class_names = ['sem_risco', 'risco_intermediario', 'alto_risco']

# Compute confusion matrix
cm = confusion_matrix(df['Label'].values, df['ROI Gated Clssf'].values, normalize='all')

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Plot the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, cmap='Blues')

plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

print('ROI Gated Clssf')
# Alternatively, print classification report
report = classification_report(df['Label'].values, df['ROI Gated Clssf'].values, target_names=class_names)
print('Classification Report:\n', report)

print('Lesion Gated Clssf')
# Alternatively, print classification report
report = classification_report(df['Label'].values, df['Lesion Gated Clssf'].values, target_names=class_names)
print('Classification Report:\n', report)

print('Circle Gated Clssf')
# Alternatively, print classification report
report = classification_report(df['Label'].values, df['Circle Gated Clssf'].values, target_names=class_names)
print('Classification Report:\n', report)

# # Plot bar graph per class comparing with label
# classes = ['sem_risco', 'risco_intermediario', 'alto_risco']
# # Setting up the positions for the bars
# bar_width = 0.2
# index = np.arange(len(classes))

# # Create the figure and axes
# fig, ax = plt.subplots(figsize=(8,6))

# # labels = df['Label'].value_counts()
# # print(labels.values)
# # 1/0
# print(df['Label'].value_counts().values)
# # print(df['Estimated Fake Gated Clssf'].value_counts())

# # Bar plots for accuracy, precision, recall, and F1-score
# ax.bar(index, df['Label'].value_counts().values, bar_width, label='Planilha')
# ax.bar(index + bar_width, df['ROI Gated Clssf'].value_counts().values, bar_width, label='ROI Gated')
# ax.bar(index + 2 * bar_width, df['Lesion Gated Clssf'].value_counts().values, bar_width, label='Lesion Gated')
# # ax.bar(index + 2 * bar_width, df['Estimated Fake Gated Clssf'].value_counts().values, bar_width, label='Estimated Fake Gated')
# # ax.bar(index + 3 * bar_width, f1_score, bar_width, label='F1 Score')

# # Adding labels and title
# ax.set_xlabel('Classes')
# ax.set_ylabel('Counts')
# ax.set_title('Comparison of Classification for 3 Classes')
# ax.set_xticks(index + bar_width * 1.5)
# ax.set_xticklabels(classes)
# ax.legend()

# # Display the plot
# plt.tight_layout()
# plt.show()
