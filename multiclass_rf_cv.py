# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np
import seaborn as sns

match_file_path = '/home/moumi/ARG-Network_analysis/protein_lists/PA_phy_deepARGDB_matches.tsv'
feature_file_path = '/home/moumi/ARG-Network_analysis/networks/physical_network/PA_gen_features.csv'
multi_gen_feature_path = '/home/moumi/ARG-Network_analysis/databases/PA_phy_multi_gen_features.csv'

df = pd.read_csv(match_file_path, delimiter='\t', header=None)

df.rename(columns={0: 'name'}, inplace=True)
df['drug_class'] = df.iloc[:, 1].apply(lambda x: x.split('|')[1].split(';')[0].split(':')[0])
df['drug_class'] = df['drug_class'].replace('beta-lactam', 'betalactam')
df['drug_class'] = df['drug_class'].replace('beta_lactam', 'betalactam')
df['drug_class'] = df['drug_class'].replace('', 'unclassified')
df['drug_class'] = df['drug_class'].replace('multidrug_efflux_SMR_transporter', 'multidrug')
df['drug_class'] = df['drug_class'].replace('macrolide-lincosamide-streptogramin', 'macrolide')
df['drug_class'] = df['drug_class'].replace('bleomycin_family_antibiotic_N-acetyltransferase', 'bleomycin')

result = df.groupby('name')['drug_class'].value_counts().groupby(level=0).idxmax().reset_index()
result['drug_class'] = result['drug_class'].str.get(1)
result['drug_ID'] = pd.factorize(result['drug_class'].values, sort=True)[0] + 1
df_features = pd.read_csv(feature_file_path)

df_nodes = df_features.merge(result, on='name', how='left')
df_nodes['drug_class'].fillna('nonARG', inplace=True)
df_nodes['drug_ID'].fillna(0, inplace=True)
df_nodes['drug_ID'] = df_nodes['drug_ID'].astype(int)

df_nodes.to_csv(multi_gen_feature_path, index=False)
new_cols = ['name','AverageShortestPathLength','BetweennessCentrality','ClosenessCentrality','ClusteringCoefficient','Degree','Eccentricity','NeighborhoodConnectivity','Radiality','Stress','TopologicalCoefficient','drug_class','drug_ID']
df_nodes = df_nodes[new_cols]
df_nodes_original = df_nodes.copy(deep=True)
mapping = dict(zip(df_nodes_original.drug_ID,df_nodes_original.drug_class))
sorted_mapping = {k: v for k, v in sorted(mapping.items())}

df_others = df_nodes[df_nodes.drug_ID == 0]
df_ARGs = df_nodes[df_nodes.drug_ID != 0]

df_ARGs_copy = df_ARGs.copy()
df_others_copy = df_others.copy()

num_ARGs = len(df_ARGs.index)
print('number of ARGs found: ', num_ARGs)
num_iterations = len(df_others.index) // num_ARGs

total_accuracy = []
total_precision = []
total_recall = []
total_fscore = []

cm_all = np.zeros((len(sorted_mapping), len(sorted_mapping)))
unique_drug_IDs = set()
model = RandomForestClassifier(n_estimators = 10, random_state = 20)

for j in range(5):
    print('outer iteration: ',j)
    df_ARGs = df_ARGs_copy
    df_others = df_others_copy

    df_ARGs_test = df_ARGs.sample(int(num_ARGs*0.2), replace=False)
    df_others_test = df_others.sample(len(df_ARGs_test.index), replace=False)
    df_test = pd.concat([df_ARGs_test, df_others_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    x_ts = df_test.values[:, 1:-2]
    y_ts = df_test.values[:, -1]
    y_ts = y_ts.astype('int')

    cond = df_others['name'].isin(df_others_test['name'])
    df_others = df_others.drop(df_others[cond].index)
    cond = df_ARGs['name'].isin(df_ARGs_test['name'])
    df_ARGs = df_ARGs.drop(df_ARGs[cond].index)

    num_iterations = len(df_others.index) // len(df_ARGs.index)

    preds = []
    for i in range(num_iterations):
        print('inner iteration: ',i)
        df_others_under = df_others.sample(len(df_ARGs.index), replace=False)
        cond = df_others['name'].isin(df_others_under['name'])
        df_others = df_others.drop(df_others[cond].index)
        df = pd.concat([df_ARGs, df_others_under])
        df = df.sample(frac=1).reset_index(drop=True)

        x = df.values[:, 1:-2]
        y = df.values[:, -1]
        y = y.astype('int')

        model.fit(x,y)

        pred = model.predict(x_ts)
        preds.append(pred)

    ensemble_pred, count = stats.mode(preds, axis = 0, keepdims=False)
    
    accuracy = accuracy_score(y_ts, ensemble_pred)

    cm = confusion_matrix(y_ts,ensemble_pred,labels=list(sorted_mapping.keys()))

    precision, recall, fscore, support = precision_recall_fscore_support(y_ts, ensemble_pred, average = 'weighted', zero_division = 1)
    
    total_accuracy.append(accuracy)
    total_precision.append(precision)
    total_recall.append(recall)
    total_fscore.append(fscore)

    row_sums = np.sum(cm, axis=1)

    result = np.divide(cm, row_sums[:, np.newaxis])
    result = np.nan_to_num(result, nan=0)
    cm_all = np.add(result, cm_all)

    drug_IDs_cm = np.unique(np.concatenate((y_ts, ensemble_pred)))
    unique_drug_IDs.update(drug_IDs_cm.flatten())
    
drug_classes_cm = [sorted_mapping[drug_ID] for drug_ID in unique_drug_IDs]
cm_all = cm_all/5
plt.figure(figsize=(15, 15))
sns.set(font_scale=2.2)
sns.heatmap(cm_all, xticklabels=list(sorted_mapping.values()), yticklabels=list(sorted_mapping.values()), annot=False, cmap='Reds')
plt.title('Pseudomonas aeruginosa', style = 'italic')

plt.show()

mean_precision = np.mean(total_precision)
mean_recall = np.mean(total_recall)
mean_f1_score = np.mean(total_fscore)
mean_accuracy = np.mean(total_accuracy)

print(f"Mean Accuracy: {mean_accuracy}")
print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean F1-score: {mean_f1_score}")

# print(type(cm_all))
# print(np.array(list(sorted_mapping.values())))

diagonal_values = np.diagonal(cm_all)
sorted_indices = np.argsort(diagonal_values)[::-1]
top_10_rows = cm_all[sorted_indices[:10], :][:, sorted_indices[:10]]

print(top_10_rows)
print(sorted_indices[:10])
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.7)
sns.heatmap(top_10_rows, xticklabels=np.array(list(sorted_mapping.values()))[list(sorted_indices[:10])], yticklabels=np.array(list(sorted_mapping.values()))[list(sorted_indices[:10])], annot=False, cmap='Reds')
plt.title('Pseudomonas aeruginosa', style = 'italic')
plt.show()

# %%
