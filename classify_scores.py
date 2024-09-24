import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def classify(x):
    if x > 0 and x < 100:
        return 'sem_risco'
    elif x >= 100 and x < 400:
        return 'risco_intermediario'
    elif x >= 400:
        return 'alto_risco'
    
scores_path = 'data/EXAMES/calcium_score_estimations.csv'

df = pd.read_csv(scores_path)

df['Label'] = df['Escore'].apply(lambda x: classify(x))
df['Estimated Gated Clssf'] = df['Estimated Score Gated'].apply(lambda x: classify(x))
df['Direct Gated Clssf'] = df['Direct Score Gated'].apply(lambda x: classify(x))
df['Estimated Fake Gated Clssf'] = df['Estimated Score Fake Gated'].apply(lambda x: classify(x))
df['Direct Fake Gated Clssf'] = df['Direct Score Fake Gated'].apply(lambda x: classify(x))

accuracy_eg = (df['Label'] == df['Estimated Gated Clssf']).sum() / len(df) *100
accuracy_dg = (df['Label'] == df['Direct Gated Clssf']).sum() / len(df) *100
accuracy_efg = (df['Label'] == df['Estimated Fake Gated Clssf']).sum() / len(df) *100
accuracy_dfg = (df['Label'] == df['Direct Fake Gated Clssf']).sum() / len(df) *100

print(f'Accuracy Estimated Gated: {accuracy_eg}%')
print(f'Accuracy Estimated Fake Gated: {accuracy_efg}%')
print(f'Accuracy Direct Fake Gated: {accuracy_dfg}%')
print(f'Accuracy Direct Gated: {accuracy_dg}%')

# Plot bar graph per class comparing with label

classes = ['sem_risco', 'risco_intermediario', 'alto_risco']
# Setting up the positions for the bars
bar_width = 0.2
index = np.arange(len(classes))

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8,6))

# labels = df['Label'].value_counts()
# print(labels.values)
# 1/0
print(df['Label'].value_counts().values)
print(df['Estimated Fake Gated Clssf'].value_counts())

# Bar plots for accuracy, precision, recall, and F1-score
ax.bar(index, df['Label'].value_counts().values, bar_width, label='Planilha')
ax.bar(index + bar_width, df['Estimated Gated Clssf'].value_counts().values, bar_width, label='Estimated Gated')
ax.bar(index + 2 * bar_width, df['Estimated Fake Gated Clssf'].value_counts().values, bar_width, label='Estimated Fake Gated')
# ax.bar(index + 3 * bar_width, f1_score, bar_width, label='F1 Score')

# Adding labels and title
ax.set_xlabel('Classes')
ax.set_ylabel('Counts')
ax.set_title('Comparison of Classification for 3 Classes')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(classes)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
