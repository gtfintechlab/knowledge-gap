import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *


## Get Robintrack
df_retrail = pd.read_csv('../data/exp_variables/robintrack/daily_data.csv')

# Convert to long format
df_retrail = df_retrail.melt(id_vars=['date'], var_name='TIC', value_name='number_investor')


df_retrail = df_retrail.dropna()
df_retrail['date'] = pd.to_datetime(df_retrail['date'], format="%d-%m-%Y")
df_retrail["year"] = df_retrail['date'].dt.year
df_retrail = df_retrail[["TIC", "number_investor", "year"]]
df_retrail = df_retrail.groupby(['year', 'TIC']).mean()
df_retrail = df_retrail.reset_index()

df_retrail.columns = ["year", "TIC", "mean_number_investor"]

print(df_retrail.shape)

df_retrail['normalized_mean_number_investor'] = (df_retrail['mean_number_investor'] - df_retrail['mean_number_investor'].mean()) / df_retrail['mean_number_investor'].std()


for model_name, file in [("llama_8b_3", "../data/llm_prompt_outputs_v2/llama8b_yearly_3_combined.csv"),
                    ("gpt4omini", "../data/llm_prompt_outputs_v2/gpt4o_mini_yearly_combined.csv"),
                    ("llama_70b_3", "../data/llm_prompt_outputs_v2/llama70b_yearly_3_combined.csv"), 
                    ("gpt4o", "../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv")]:

    df_combined_model = pd.read_csv(file)

    df_combined_model['extracted_revenue'] = df_combined_model['prompt_output'].apply(extract_revenue)
    df_combined_model['success'] = ~np.isnan(df_combined_model['extracted_revenue'])
    df_combined_model['ad_hoc_error'] = df_combined_model.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_model['low'] = df_combined_model['ad_hoc_error'].apply(bool_low)

    ## Merge Robintrack
    df_combined_model = df_combined_model.merge(df_retrail, on=['TIC', 'year'], how='inner')


    ##################### Run regression #####################



    # Create dummy variables for each year
    year_dummies = pd.get_dummies(df_combined_model['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(df_combined_model['normalized_mean_number_investor']), year_dummies], axis=1)

    y = df_combined_model['low']

    # Create and fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open(f'./regressions_v2/result_summary_RTrack_{model_name}.txt', "w") as f:
        f.write(str(result.summary()))
