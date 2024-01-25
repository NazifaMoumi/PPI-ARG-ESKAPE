# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
import numpy as np
import seaborn as sns

# match_file_path = ['/home/moumi/ARG-Network_analysis/protein_lists/PA_phy_deepARGDB_matches.tsv', '/home/moumi/ARG-Network_analysis/protein_lists/SA_phy_deepARGDB_matches.tsv', '/home/moumi/ARG-Network_analysis/protein_lists/KP_phy_deepARGDB_matches.tsv', '/home/moumi/ARG-Network_analysis/protein_lists/EF_phy_deepARGDB_matches.tsv', '/home/moumi/ARG-Network_analysis/protein_lists/AB_phy_deepARGDB_matches.tsv']
match_file_path = '/home/moumi/ARG-Network_analysis/protein_lists/EnC_phy_deepARGDB_matches.tsv'
feature_file_path = '/home/moumi/ARG-Network_analysis/networks/physical_network/EnC_gen_features.csv'
all_feature_file_path = '/home/moumi/ARG-Network_analysis/databases/EnC_phy_all_features.csv'
test_ARGs = '/home/moumi/ARG-Network_analysis/protein_lists/EnC_phy_test_ARGs.txt'
all_feature_files = ['/home/moumi/ARG-Network_analysis/networks/physical_network/565664_gen_features.csv', '/home/moumi/ARG-Network_analysis/networks/physical_network/1280_gen_features.csv', '/home/moumi/ARG-Network_analysis/networks/physical_network/573_gen_features.csv', '/home/moumi/ARG-Network_analysis/networks/physical_network/470_gen_features.csv', '/home/moumi/ARG-Network_analysis/networks/physical_network/287_gen_features.csv']
multi_gen_feature_path = '/home/moumi/ARG-Network_analysis/databases/EnC_phy_multi_gen_features.csv'
# test_ARGs = ['/home/moumi/ARG-Network_analysis/protein_lists/EF_phy_test_ARGs.txt', '/home/moumi/ARG-Network_analysis/protein_lists/KP_phy_test_ARGs.txt', '/home/moumi/ARG-Network_analysis/protein_lists/PA_phy_test_ARGs.txt', '/home/moumi/ARG-Network_analysis/protein_lists/SA_phy_test_ARGs.txt', '/home/moumi/ARG-Network_analysis/protein_lists/AB_phy_test_ARGs.txt']

df = pd.read_csv(match_file_path, delimiter='\t', header=None)
# df = pd.concat((pd.read_csv(f, delimiter='\t', header=None) for f in match_file_path), ignore_index=True)

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
# df_features = pd.concat((pd.read_csv(f) for f in all_feature_files), ignore_index=True)
df_features = pd.read_csv(feature_file_path)
# df_features = df_features.rename(columns = {'prot_ID':'name'})
# df_features = df_features.drop('y',axis=1)

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

test_ARG_prots = []
# for test_ARG_file in test_ARGs:
with open(test_ARGs,'r') as f:
    lines = f.readlines()
    for line in lines:
        test_ARG_prots.append(line.strip())

df_test_args = df_nodes[df_nodes['name'].isin(test_ARG_prots)]

total_accuracy = 0
total_precision = 0
total_recall = 0
total_fscore = 0

cm_all = np.zeros((len(sorted_mapping), len(sorted_mapping)))
unique_drug_IDs = set()
for j in range(10):

    df_nodes = df_nodes_original
    df_others = df_nodes[df_nodes.drug_ID == 0]

    df_test_others = df_others.sample(len(df_test_args.index), replace=False)
    df_test = pd.concat([df_test_args, df_test_others])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    x_ts = df_test.values[:, 1:-2]
    y_ts = df_test.values[:, -1]
    y_ts = y_ts.astype('int')

    cond = df_nodes['name'].isin(df_test['name'])
    df_nodes = df_nodes.drop(df_nodes[cond].index)
    df_train_args = df_nodes[df_nodes.drug_ID != 0]
    df_train_others = df_nodes[df_nodes.drug_ID == 0]

    num_models = len(df_train_others.index)//len(df_train_args.index)

    print('num models: ',num_models)
    model = RandomForestClassifier(n_estimators = 10, random_state = 20)

    preds = []

    for i in range(num_models):
        df_train_others_sampled = df_train_others.sample(len(df_train_args.index), replace=False)
        df_train_others = df_train_others.drop(df_train_others_sampled.index)
        df_nodes = pd.concat([df_train_args, df_train_others_sampled])
        df_nodes = df_nodes.sample(frac=1).reset_index(drop=True)

        x = df_nodes.values[:, 1:-2]
        y = df_nodes.values[:, -1]
        y = y.astype('int')

        model.fit(x,y)

        pred = model.predict(x_ts)
        preds.append(pred)

    ensemble_pred, count = stats.mode(preds, axis = 0, keepdims=False)
    
    accuracy = accuracy_score(y_ts, ensemble_pred)

    cm = confusion_matrix(y_ts,ensemble_pred,labels=list(sorted_mapping.keys()))
    print('cm length ',len(cm))

    precision, recall, fscore, support = precision_recall_fscore_support(y_ts, ensemble_pred, average = 'weighted', zero_division = 1)
    
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_fscore += fscore

    row_sums = np.sum(cm, axis=1)

    result = np.divide(cm, row_sums[:, np.newaxis])
    result = np.nan_to_num(result, nan=0)
    cm_all = np.add(result, cm_all)

    drug_IDs_cm = np.unique(np.concatenate((y_ts, ensemble_pred)))
    unique_drug_IDs.update(drug_IDs_cm.flatten())
    
drug_classes_cm = [sorted_mapping[drug_ID] for drug_ID in unique_drug_IDs]
cm_all = cm_all/10
print(cm_all)
sns.heatmap(cm_all, xticklabels=list(sorted_mapping.values()), yticklabels=list(sorted_mapping.values()), annot=False, cmap='Reds')
plt.show()

# combined_arr = np.vstack(ensemble_pred_list)

# result = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=combined_arr)
# cm = confusion_matrix(y_ts,result)
# print(cm)
print('accuracy: ',total_accuracy/10)
print('precision: ',total_precision/10)
print('recall: ',total_recall/10)
print('fscore: ',total_fscore/10)
# %%
