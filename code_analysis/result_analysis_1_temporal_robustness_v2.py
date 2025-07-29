import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *

############## for ChatGPT

df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv')


df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])
df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
df_combined_chatgpt['low'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low)
df_combined_chatgpt['low_5'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low_5)
df_combined_chatgpt['low_20'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low_20)


# Group by 'year' and calculate the percentage of rows where 'column' equals 'low'
chatgpt_result = df_combined_chatgpt.groupby('year').apply(lambda x: calculate_std_per(x, 'chatgpt')).reset_index()


# print(chatgpt_result)


# Plotting
plt.plot(chatgpt_result.year, chatgpt_result.chatgpt_correctness_5, color='r', label='5%')
plt.fill_between(chatgpt_result.year, (chatgpt_result.chatgpt_correctness_5-chatgpt_result.chatgpt_std_5), \
                                        (chatgpt_result.chatgpt_correctness_5+chatgpt_result.chatgpt_std_5), color='r', alpha=0.1)

plt.plot(chatgpt_result.year, chatgpt_result.chatgpt_correctness_10, color='b', label='10%')
plt.fill_between(chatgpt_result.year, (chatgpt_result.chatgpt_correctness_10-chatgpt_result.chatgpt_std_10), \
                                        (chatgpt_result.chatgpt_correctness_10+chatgpt_result.chatgpt_std_10), color='b', alpha=0.1)

plt.plot(chatgpt_result.year, chatgpt_result.chatgpt_correctness_20, color='darkorange', label='20%')
plt.fill_between(chatgpt_result.year, (chatgpt_result.chatgpt_correctness_20-chatgpt_result.chatgpt_std_20), \
                                        (chatgpt_result.chatgpt_correctness_20+chatgpt_result.chatgpt_std_20), color='darkorange', alpha=0.1)


# Add vertical dotted line at year 1995
plt.axvline(1995, color='grey', linestyle='--', linewidth=1.5)

# Adding labels
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Perecentage %', fontweight='bold')
plt.title('Success Rate of GPT-4o with Different Error Threshold')
plt.legend(loc='upper left') 

plt.rcParams['font.size'] = 25

# Show the plot
plt.tight_layout()
plt.show()