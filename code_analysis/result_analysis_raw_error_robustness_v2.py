import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl

def check_answer(true_rev, extracted_rev, success):
    if success != False:
        if true_rev == 0.0:
            return None
        else:
            return abs((extracted_rev-true_rev)/true_rev)*100
    return None


if __name__ == '__main__':

    ## ============================== ChatGPT ==================================
    df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv')

    df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
    df_combined_chatgpt_2010 = df_combined_chatgpt[df_combined_chatgpt['year'] == 2010]

    df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])

    success_count_per_year = df_combined_chatgpt.groupby('year')['success'].sum()
    attempt_count_per_year = df_combined_chatgpt.groupby('year')['success'].count()
    success_percentage_per_year = (success_count_per_year / attempt_count_per_year) * 100

    # Now plot the result
    success_percentage_per_year.plot(kind='line', marker='o')

    plt.title('Percentage of Successes per Year')
    plt.xlabel('Year')
    plt.ylabel('Success Percentage')
    plt.xticks(rotation=0) 
    # plt.savefig('success_plot.png')
    plt.close()

    plt.figure(figsize=(55, 18))

    import seaborn as sns
    import matplotlib.pyplot as plt

    df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: check_answer(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    total_data_points_per_year = df_combined_chatgpt.groupby('year')['ad_hoc_error'].size()
    df_combined_chatgpt = df_combined_chatgpt.dropna(subset=['ad_hoc_error'])

    data_points_per_year = df_combined_chatgpt.groupby('year')['ad_hoc_error'].size()
    data_percent_per_year = data_points_per_year / total_data_points_per_year
    print(data_percent_per_year)
    normalized_data_points = (data_points_per_year - data_points_per_year.min()) / \
                            (data_points_per_year.max() - data_points_per_year.min())
    
    norm = mpl.colors.Normalize(vmin=min(data_percent_per_year), vmax=max(data_percent_per_year))
    cmap = plt.cm.coolwarm
    colors = cmap(data_percent_per_year)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    Q1 = df_combined_chatgpt['ad_hoc_error'].quantile(0.25)
    Q3 = df_combined_chatgpt['ad_hoc_error'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_combined_chatgpt[(df_combined_chatgpt['ad_hoc_error'] >= lower_bound) & (df_combined_chatgpt['ad_hoc_error'] <= upper_bound)]
    median = df_filtered.groupby('year')['ad_hoc_error'].median()
    
    import seaborn as sns
    plt.figure(figsize=(40, 24))
    for year, color in zip(data_points_per_year.index, colors):
        sns.boxplot(x='year', y='ad_hoc_error', data=df_filtered[df_filtered['year'] == year], color=color, boxprops=dict(alpha=.5))

    plt.plot(range(0, len(median)), median, color='red', linestyle='-', marker='o')
    cbar = plt.colorbar(sm, orientation='vertical', fraction=0.03, pad=0.03, aspect=20)
    cbar.set_label("GPT-4o's Output Success Rate", rotation=270, labelpad=30, fontsize=20)
    cbar.set_ticks(np.linspace(min(data_percent_per_year), max(data_percent_per_year), num=len(data_percent_per_year)))
    data_percent_per_year = data_percent_per_year.round(4)
    data_percent_per_year *= 100
    data_percent_per_year = data_percent_per_year.round(4)
    cbar.set_ticklabels(sorted(data_percent_per_year.values), fontsize=15)


    plt.title('Raw Error by Year', fontsize=50)
    plt.xlabel('Year', fontsize=30)
    plt.ylabel('Raw Error Percentage', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    # plt.show()
    plt.savefig('chatgpt_raw_error.png', dpi=300)

