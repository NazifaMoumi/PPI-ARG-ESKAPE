# %%

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def create_matrix():
    organisms = ['EF', 'SA', 'KP', 'AB', 'PA', 'EnC']
    merged_df = None
    for organism in organisms:
        feature_file_path = '/home/moumi/ARG-Network_analysis/databases/' + organism + '_phy_multi_gen_features.csv'
        df = pd.read_csv(feature_file_path)
        drug_counts = df['drug_class'].value_counts()
        df = pd.DataFrame({'drug_class':drug_counts.index, organism:drug_counts.values})
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=merged_df.columns[0], how='outer')
    merged_df = merged_df.fillna(0)
    return merged_df

def visualize(merged_df):
    merged_df = merged_df[merged_df.drug_class != 'nonARG']
    merged_df = merged_df.set_index('drug_class')
    ax = merged_df.plot.bar(stacked=True)
    plt.show()

visualize(create_matrix())
# %%
