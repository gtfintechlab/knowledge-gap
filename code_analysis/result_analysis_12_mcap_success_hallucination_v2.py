import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from utils import *

# Function to categorize mcap
def categorize_mcap(value):
    if value < 8.0:
        return '<8.0'
    elif 8.0 <= value < 9.0:
        return '8.0-9.0'
    elif 9.0 <= value < 10.0:
        return '9.0-10.0'
    else:
        return '>=10.0'

for model_name, file in [("gpt4o", "../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv"),
                         ("llama_70b_3", "../data/llm_prompt_outputs_v2/llama70b_yearly_3_combined.csv")]:

    df_combined_model = pd.read_csv(file)

    df_combined_model['extracted_revenue'] = df_combined_model['prompt_output'].apply(extract_revenue)
    df_combined_model['success'] = ~np.isnan(df_combined_model['extracted_revenue'])
    df_combined_model['ad_hoc_error'] = df_combined_model.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_model['low'] = df_combined_model['ad_hoc_error'].apply(bool_low)


    df_cpi_factors = pd.read_csv("../data/CPI_factors.csv")
    df_combined_model = df_combined_model.merge(df_cpi_factors, on='year', how='left')
    df_combined_model["adj_mcap"] = df_combined_model["mcap"] + np.log10(df_combined_model["CPI_factor"])


    df_try = df_combined_model

    df_try = get_same_comp(df_try, 1980, 2020)

    # grayscale plot mean MCap

    print(df_try.head())

    df_try = df_try.groupby('GVKEY').apply(lambda x: pd.DataFrame({'chatgpt_success': [(x['low'] == True).sum()], 
                                                                'chatgpt_hal': [((x['success'] == True) & (x['low'] == False)).sum()],
                                                                'avg_mcap': [x['adj_mcap'].mean()]})).reset_index(level=0).reset_index(drop=True)

    # Apply categorization
    df_try['mean_mcap_category'] = df_try['avg_mcap'].apply(categorize_mcap)

    plt.rcParams['font.size'] = 10

    # scatter = plt.scatter(df_try.chatgpt_success, df_try.chatgpt_hal, c=df_try.avg_mcap, cmap='Greys')
    scatter = plt.scatter(df_try.chatgpt_success, df_try.chatgpt_hal, c=df_try.avg_mcap, cmap='viridis')

    # Add a color bar
    plt.colorbar(scatter, label='MCap (log)')

    plt.xlabel('Success Count (# Years)')
    plt.ylabel('Hallucination Count (# Years)')
    if model_name=="gpt4o":
        plt.title('Hallucination Analysis for GPT-4o')
    else:
        plt.title('Hallucination Analysis for LLaMA-3-70B-Chat')
    plt.show()
