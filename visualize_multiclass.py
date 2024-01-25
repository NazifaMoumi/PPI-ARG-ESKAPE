# %%
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

def compute_metrics(organism):
    feature_file_path = '/home/moumi/ARG-Network_analysis/databases/' + organism + '_phy_all_features.csv'
    test_ARG_path = '/home/moumi/ARG-Network_analysis/protein_lists/' + organism + '_phy_test_ARGs.txt'    
    df_nodes = pd.read_csv(feature_file_path)

    df_others = df_nodes[df_nodes.y == 0]

    test_ARG_prots = []
    with open(test_ARG_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            test_ARG_prots.append(line.strip())

    df_test_args = df_nodes[df_nodes['prot_ID'].isin(test_ARG_prots)]

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_fscore = 0

    for j in range(10):
        df_nodes = pd.read_csv(feature_file_path)
        df_others = df_nodes[df_nodes.y == 0]

        df_test_others = df_others.sample(len(df_test_args.index), replace=False)
        df_test = pd.concat([df_test_args, df_test_others])
        df_test = df_test.sample(frac=1).reset_index(drop=True)

        x_ts = df_test.values[:, 1:-1]
        y_ts = df_test.values[:, -1]
        y_ts = y_ts.astype('int')

        cond = df_nodes['prot_ID'].isin(df_test['prot_ID'])
        df_nodes.drop(df_nodes[cond].index, inplace = True)

        df_train_args = df_nodes[df_nodes.y == 1]
        df_train_others = df_nodes[df_nodes.y == 0]

        num_models = len(df_train_others.index)//len(df_train_args.index)

        # print('num models: ',num_models)
        model = RandomForestClassifier(n_estimators = 10, random_state = 20)

        preds = []
        for i in range(num_models):
            df_train_others_sampled = df_train_others.sample(len(df_train_args.index), replace=False)
            df_train_others = df_train_others.drop(df_train_others_sampled.index)
            df_nodes = pd.concat([df_train_args, df_train_others_sampled])

            x = df_nodes.values[:, 1:-1]
            y = df_nodes.values[:, -1]
            y=y.astype('int')

            model.fit(x,y)

            pred = model.predict(x_ts)
            preds.append(pred)

        ensemble_pred, count = stats.mode(preds, axis = 0, keepdims=False)
        accuracy = accuracy_score(y_ts, ensemble_pred)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(y_ts, ensemble_pred, average = 'binary')
        
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_fscore += fscore

    return (total_accuracy/10, total_precision/10, total_recall/10, total_fscore/10)

def main():
    organisms = ['EF', 'SA', 'KP', 'AB', 'PA']
    precisions = []
    recalls = []
    accuracies = []
    fscores = []

    multiclass_results = {'EF':[0.7444444444444445, 0.8003122440622441, 0.7444444444444445, 0.7095588393657681],'SA':[0.7, 0.7772788955272605, 0.7,  0.662677433618095],'KP':[0.7118279569892474, 0.7316681497898125, 0.7118279569892474, 0.6736870894707014],'AB':[0.6406976744186048, 0.7200320257973406, 0.6406976744186048, 0.5870459083907706],'PA':[0.6933112649274774, 0.6617977528089887, 0.6933112649274774, 0.6617977528089887]}
    
    for organism in organisms:
        accuracies.append(multiclass_results[organism][0])
        precisions.append(multiclass_results[organism][1])
        recalls.append(multiclass_results[organism][2])
        fscores.append(multiclass_results[organism][3])

    data = {
        'Organism': organisms,
        'Precision': precisions,
        'Recall': recalls,
        'Accuracy': accuracies,
        'F1-score': fscores
    }

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Organism', var_name='Metric', value_name='Score')

    y1 = df_melted.Score

    fig, ax = plt.subplots()

    sns.barplot(x='Organism', y=y1, hue='Metric', data=df_melted, alpha=1, ax=ax)
    plt.legend(title='Metrics')
    
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[:4]  # Keep only the handles for the first set of bars
    labels = labels[:4]  # Keep only the labels for the first set of bars
    ax.legend(handles, labels, frameon=True, fancybox=True, edgecolor="black", facecolor='white', loc='lower right')
    plt.show()
        
if __name__ == "__main__":
    main()
# %%
