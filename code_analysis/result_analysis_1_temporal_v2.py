import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *

############## for LLaMA

df_combined_llama = pd.read_csv('../data/llm_prompt_outputs_v2/llama8b_yearly_3_combined.csv')


df_combined_llama['extracted_revenue'] = df_combined_llama['prompt_output'].apply(extract_revenue)
df_combined_llama['success'] = ~np.isnan(df_combined_llama['extracted_revenue'])
df_combined_llama['ad_hoc_error'] = df_combined_llama.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_llama['low'] = df_combined_llama['ad_hoc_error'].apply(bool_low)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
llama_result = df_combined_llama.groupby('year').apply(lambda x: calculate_std(x, 'llama')).reset_index()

############## for LLaMA 70B

df_combined_llama = pd.read_csv('../data/llm_prompt_outputs_v2/llama70b_yearly_3_combined.csv')


df_combined_llama['extracted_revenue'] = df_combined_llama['prompt_output'].apply(extract_revenue)
df_combined_llama['success'] = ~np.isnan(df_combined_llama['extracted_revenue'])
df_combined_llama['ad_hoc_error'] = df_combined_llama.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_llama['low'] = df_combined_llama['ad_hoc_error'].apply(bool_low)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
llama_result_70b = df_combined_llama.groupby('year').apply(lambda x: calculate_std(x, 'llama_70b')).reset_index()

print(llama_result_70b)

############## for GPT4o-mini

df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_mini_yearly_combined.csv')


df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])
df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_chatgpt['low'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
chatgpt_result = df_combined_chatgpt.groupby('year').apply(lambda x: calculate_std(x, 'gpt4omini')).reset_index()


############## for GPT4o

df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv')


df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])
df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_chatgpt['low'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
chatgpt_result_4o = df_combined_chatgpt.groupby('year').apply(lambda x: calculate_std(x, 'gpt4o')).reset_index()




combined_result = pd.merge(llama_result_70b, chatgpt_result, on="year", how='left')
combined_result = pd.merge(combined_result, llama_result, on="year", how='left')
combined_result = pd.merge(combined_result, chatgpt_result_4o, on="year", how='left')

print(combined_result)


# Plotting
plt.plot(combined_result.year, combined_result.llama_correctness, color='r', label='LLaMA-3-8B-Chat')
plt.fill_between(combined_result.year, (combined_result.llama_correctness-combined_result.llama_std), \
                                        (combined_result.llama_correctness+combined_result.llama_std), color='r', alpha=0.1)

plt.plot(combined_result.year, combined_result.llama_70b_correctness, color='darkorange', label='LLaMA-3-70B-Chat')
plt.fill_between(combined_result.year, (combined_result.llama_70b_correctness-combined_result.llama_70b_std), \
                                        (combined_result.llama_70b_correctness+combined_result.llama_70b_std), color='darkorange', alpha=0.1)

plt.plot(combined_result.year, combined_result.gpt4omini_correctness, color='b', label='GPT-4o-mini')
plt.fill_between(combined_result.year, (combined_result.gpt4omini_correctness-combined_result.gpt4omini_std), \
                                        (combined_result.gpt4omini_correctness+combined_result.gpt4omini_std), color='b', alpha=0.1)

plt.plot(combined_result.year, combined_result.gpt4o_correctness, color='m', label='GPT-4o')
plt.fill_between(combined_result.year, (combined_result.gpt4o_correctness-combined_result.gpt4o_std), \
                                        (combined_result.gpt4o_correctness+combined_result.gpt4o_std), color='b', alpha=0.1)

# Add vertical dotted line at year 1995
plt.axvline(1995, color='grey', linestyle='--', linewidth=1.5)

# Adding labels
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Perecentage %', fontweight='bold')
plt.title('Success Rate For Different Models')
plt.legend(loc='upper left') 

plt.rcParams['font.size'] = 25

# Show the plot
plt.tight_layout()
plt.show()