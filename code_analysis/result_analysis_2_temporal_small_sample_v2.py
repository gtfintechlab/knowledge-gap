import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *

df_combined_gpt45 = pd.read_csv('../data/llm_prompt_outputs_v2/gpt45_yearly_combined_small.csv')


df_combined_gpt45['extracted_revenue'] = df_combined_gpt45['prompt_output'].apply(extract_revenue)
df_combined_gpt45['success'] = ~np.isnan(df_combined_gpt45['extracted_revenue'])
df_combined_gpt45['ad_hoc_error'] = df_combined_gpt45.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_gpt45['low'] = df_combined_gpt45['ad_hoc_error'].apply(bool_low)


gpt45_result = df_combined_gpt45.groupby('year').apply(lambda x: calculate_std(x, 'gpt45')).reset_index()


df_combined_deepseek = pd.read_csv('../data/llm_prompt_outputs_v2/deepseek_v3_yearly_combined_small.csv')


df_combined_deepseek['extracted_revenue'] = df_combined_deepseek['prompt_output'].apply(extract_revenue)
df_combined_deepseek['success'] = ~np.isnan(df_combined_deepseek['extracted_revenue'])
df_combined_deepseek['ad_hoc_error'] = df_combined_deepseek.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_deepseek['low'] = df_combined_deepseek['ad_hoc_error'].apply(bool_low)

###########

# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
deepseek_result = df_combined_deepseek.groupby('year').apply(lambda x: calculate_std(x, 'deepseek')).reset_index()





df_combined_gemini15pro = pd.read_csv('../data/llm_prompt_outputs_v2/gemini15_yearly_combined_small.csv')


df_combined_gemini15pro['extracted_revenue'] = df_combined_gemini15pro['prompt_output'].apply(extract_revenue)
df_combined_gemini15pro['success'] = ~np.isnan(df_combined_gemini15pro['extracted_revenue'])
df_combined_gemini15pro['ad_hoc_error'] = df_combined_gemini15pro.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_gemini15pro['low'] = df_combined_gemini15pro['ad_hoc_error'].apply(bool_low)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
gemini15pro_result = df_combined_gemini15pro.groupby('year').apply(lambda x: calculate_std(x, 'gemini15pro')).reset_index()





df_combined_finma = pd.read_csv('../data/llm_prompt_outputs_v2/finma_7b_yearly_combined_small.csv')


df_combined_finma['extracted_revenue'] = df_combined_finma['prompt_output'].apply(extract_revenue)
df_combined_finma['success'] = ~np.isnan(df_combined_finma['extracted_revenue'])
df_combined_finma['ad_hoc_error'] = df_combined_finma.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_finma['low'] = df_combined_finma['ad_hoc_error'].apply(bool_low)

############### code to only keep same firm-year for comparison
df_combined_finma = pd.merge(df_combined_finma, df_combined_gemini15pro[['GVKEY', 'year']], on=['GVKEY', 'year'], how='inner')

###########

# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
finma_result = df_combined_finma.groupby('year').apply(lambda x: calculate_std(x, 'finma')).reset_index()





combined_result = pd.merge(gpt45_result, deepseek_result, on="year", how='left')
combined_result = pd.merge(combined_result, gemini15pro_result, on="year", how='left')
combined_result = pd.merge(combined_result, finma_result, on="year", how='left')


# Plotting
plt.plot(combined_result.year, combined_result.gpt45_correctness, color='m', label='GPT-4.5')
plt.fill_between(combined_result.year, (combined_result.gpt45_correctness-combined_result.gpt45_std), \
                                        (combined_result.gpt45_correctness+combined_result.gpt45_std), color='m', alpha=0.1)

plt.plot(combined_result.year, combined_result.deepseek_correctness, color='b', label='DeepSeek V3')
plt.fill_between(combined_result.year, (combined_result.deepseek_correctness-combined_result.deepseek_std), \
                                        (combined_result.deepseek_correctness+combined_result.deepseek_std), color='b', alpha=0.1)

plt.plot(combined_result.year, combined_result.gemini15pro_correctness, color='darkorange', label='Gemini 1.5 Pro')
plt.fill_between(combined_result.year, (combined_result.gemini15pro_correctness-combined_result.gemini15pro_std), \
                                        (combined_result.gemini15pro_correctness+combined_result.gemini15pro_std), color='darkorange', alpha=0.1)

plt.plot(combined_result.year, combined_result.finma_correctness, color='r', label='FinMA-7B-full')
plt.fill_between(combined_result.year, (combined_result.finma_correctness-combined_result.finma_std), \
                                        (combined_result.finma_correctness+combined_result.finma_std), color='r', alpha=0.1)
# Add vertical dotted line at year 1995
plt.axvline(1995, color='grey', linestyle='--', linewidth=1.5)

# Adding labels
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Perecentage %', fontweight='bold')
plt.title('Success Rate For Different Models (Small Sample)')
plt.legend(loc='upper left') 

plt.rcParams['font.size'] = 25

# Show the plot
plt.tight_layout()
plt.show()