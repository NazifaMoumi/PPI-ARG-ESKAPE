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
    organisms = ['EF', 'SA', 'KP', 'AB', 'PA', 'EnC']
    precisions = []
    recalls = []
    accuracies = []
    fscores = []
    precisions_random = []
    recalls_random = []
    accuracies_random = []
    fscores_random = []
    random_positive_results = {'EF':[0.4854166666666667, 0.47730187087597925, 0.3433333333333334, 0.3962180436284544],'SA':[0.4873611111111111, 0.4811754060785883, 0.34833333333333333,  0.40086280686811826],'KP':[0.4881720430107527, 0.47933065212231013, 0.28419354838709676, 0.3553860275475156],'AB':[0.5126744186046512, 0.5156289827251577, 0.38534883720930235, 0.4379308362944981],'PA':[0.5051123595505618, 0.5069194118815692, 0.3716853932584269, 0.42786007805607823], 'EnC':[0.5164285714285715, 0.5212216253073183, 0.40038961038961035, 0.45140196721146675]}
    
    for organism in organisms:
        results = compute_metrics(organism)
        accuracies.append(results[0])
        precisions.append(results[1])
        recalls.append(results[2])
        fscores.append(results[3])

        accuracies_random.append(random_positive_results[organism][0])
        precisions_random.append(random_positive_results[organism][1])
        recalls_random.append(random_positive_results[organism][2])
        fscores_random.append(random_positive_results[organism][3])

    data = {
        'Organism': organisms,
        'Precision': precisions,
        'Recall': recalls,
        'Accuracy': accuracies,
        'F1-score': fscores
    }
    data_random = {
        'Organism': organisms,
        'Precision': precisions_random,
        'Recall': recalls_random,
        'Accuracy': accuracies_random,
        'F1-score': fscores_random
    }
    df = pd.DataFrame(data)
    df_random = pd.DataFrame(data_random)
    df_melted = df.melt(id_vars='Organism', var_name='Metric', value_name='Score')
    df_melted_random = df_random.melt(id_vars='Organism', var_name='Metric', value_name='Score')

    y1 = df_melted.Score
    y2 = df_melted_random.Score

    # fig, ax = plt.subplots()

    # sns.barplot(x='Organism', y=y1, hue='Metric', data=df_melted, alpha=1, ax=ax)
    # plt.legend(title='Metrics')
    # sns.barplot(x='Organism', y=y2, hue='Metric', color = 'grey', data=df_melted_random, alpha=0.5, ax=ax)
    
    # handles, labels = ax.get_legend_handles_labels()
    # handles = handles[:4]  # Keep only the handles for the first set of bars
    # labels = labels[:4]  # Keep only the labels for the first set of bars
    # ax.legend(handles, labels, frameon=True, fancybox=True, edgecolor="black", facecolor='white', loc='lower right')
    # plt.show()

    fig, ax = plt.subplots()

    sns.barplot(x='Organism', y=y1, hue='Metric', data=df_melted, alpha=1, ax=ax)

    marker = '*'

    for i, bar in enumerate(ax.patches):
        value_random = y2[i]
        x_pos = bar.get_x() + bar.get_width() / 2
        ax.text(x_pos, value_random, marker, color='black', ha='center', va='bottom', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()

    handles.append(plt.Line2D([0], [0], marker=marker, color='black', markersize=10, linestyle='None'))
    labels.append('Random Positive')

    ax.legend(handles, labels, loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
# %%
