import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *


df_cpi_factors = pd.read_csv("../data/CPI_factors.csv")

for model_name, file in [("deepseek", '../data/llm_prompt_outputs_v2/deepseek_v3_yearly_combined_small.csv'),
                    ("gemini15pro", '../data/llm_prompt_outputs_v2/gemini15_yearly_combined_small.csv'),
                    ("gpt45", '../data/llm_prompt_outputs_v2/gpt45_yearly_combined_small.csv')]:

    df_combined_model = pd.read_csv(file)

    df_combined_model['extracted_revenue'] = df_combined_model['prompt_output'].apply(extract_revenue)
    df_combined_model['success'] = ~np.isnan(df_combined_model['extracted_revenue'])
    df_combined_model['ad_hoc_error'] = df_combined_model.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_model['low'] = df_combined_model['ad_hoc_error'].apply(bool_low)

    df_combined_model['hal'] = df_combined_model.apply(lambda x: ((x['success'] == True) & (x['low'] == False)), axis=1)


    df_combined_model = df_combined_model.merge(df_cpi_factors, on='year', how='left')
    df_combined_model["adj_mcap"] = df_combined_model["mcap"] + np.log10(df_combined_model["CPI_factor"])

    ##################### Run regression #####################



    # Create dummy variables for each year
    year_dummies = pd.get_dummies(df_combined_model['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(df_combined_model['adj_mcap']), year_dummies], axis=1)

    y = df_combined_model['hal']
    

    # Create and fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open(f'./regressions_v2/hal_result_summary_mcap_{model_name}.txt', "w") as f:
        f.write(str(result.summary()))
