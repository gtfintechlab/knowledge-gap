import pandas as pd
import numpy as np
from utils import *
import statsmodels.api as sm

if __name__ == '__main__':
    ## ============================== LLaMA ===================================
    df_cpi_factors = pd.read_csv("../data/CPI_factors.csv")
    df_combined_llama = pd.read_csv('../data/llm_prompt_outputs_v2/llama8b_yearly_3_combined.csv')


    df_combined_llama['extracted_revenue'] = df_combined_llama['prompt_output'].apply(extract_revenue)
    df_combined_llama['success'] = ~np.isnan(df_combined_llama['extracted_revenue'])
    df_combined_llama['ad_hoc_error'] = df_combined_llama.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_llama['low'] = df_combined_llama['ad_hoc_error'].apply(bool_low)

    df_combined_llama = df_combined_llama.merge(df_cpi_factors, on='year', how='left')
    df_combined_llama["adj_mcap"] = df_combined_llama["mcap"] + np.log10(df_combined_llama["CPI_factor"])
    
    same_comp_df_llama = get_same_comp(df_combined_llama, 1980, 2022)
    unique_comp_llama = same_comp_df_llama.drop_duplicates(subset='GVKEY', keep='last')
    number_unique_comp = unique_comp_llama.shape[0]

    # Create dummy variables for each year
    year_dummies = pd.get_dummies(same_comp_df_llama['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(same_comp_df_llama['adj_mcap']), year_dummies], axis=1)

    y = same_comp_df_llama['low']

    # Create and fit the logistic regression model
    model = sm.Logit(y, X.astype(float))
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open("./regressions_v2/result_summary_same_mcap_llama_8b.txt", "w") as f:
        f.write(str(result.summary()))

    ## ============================== LLaMA-70B ===================================
    df_cpi_factors = pd.read_csv("../data/CPI_factors.csv")
    df_combined_llama = pd.read_csv('../data/llm_prompt_outputs_v2/llama70b_yearly_3_combined.csv')


    df_combined_llama['extracted_revenue'] = df_combined_llama['prompt_output'].apply(extract_revenue)
    df_combined_llama['success'] = ~np.isnan(df_combined_llama['extracted_revenue'])
    df_combined_llama['ad_hoc_error'] = df_combined_llama.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_llama['low'] = df_combined_llama['ad_hoc_error'].apply(bool_low)

    df_combined_llama = df_combined_llama.merge(df_cpi_factors, on='year', how='left')
    df_combined_llama["adj_mcap"] = df_combined_llama["mcap"] + np.log10(df_combined_llama["CPI_factor"])
    
    same_comp_df_llama = get_same_comp(df_combined_llama, 1980, 2022)
    unique_comp_llama = same_comp_df_llama.drop_duplicates(subset='GVKEY', keep='last')
    number_unique_comp = unique_comp_llama.shape[0]

    # Create dummy variables for each year
    year_dummies = pd.get_dummies(same_comp_df_llama['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(same_comp_df_llama['adj_mcap']), year_dummies], axis=1)

    y = same_comp_df_llama['low']

    # Create and fit the logistic regression model
    model = sm.Logit(y, X.astype(float))
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open("./regressions_v2/result_summary_same_mcap_llama_70b.txt", "w") as f:
        f.write(str(result.summary()))

    ## ============================== GPT4o-mini ==================================
    df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_mini_yearly_combined.csv')


    df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
    df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])
    df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_chatgpt['low'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low)
    df_combined_chatgpt = df_combined_chatgpt.merge(df_cpi_factors, on='year', how='left')
    df_combined_chatgpt["adj_mcap"] = df_combined_chatgpt["mcap"] + np.log10(df_combined_llama["CPI_factor"])

    same_comp_df_chatgpt = get_same_comp(df_combined_chatgpt, 1980, 2022)

    # Create dummy variables for each year
    year_dummies = pd.get_dummies(same_comp_df_chatgpt['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(same_comp_df_chatgpt['adj_mcap']), year_dummies], axis=1)

    y = same_comp_df_chatgpt['low']

    # Create and fit the logistic regression model
    model = sm.Logit(y, X.astype(float))
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open("./regressions_v2/result_summary_same_mcap_gpt4o_mini.txt", "w") as f:
        f.write(str(result.summary()))

    ## ============================== GPT4o ==================================
    df_combined_chatgpt = pd.read_csv('../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv')


    df_combined_chatgpt['extracted_revenue'] = df_combined_chatgpt['prompt_output'].apply(extract_revenue)
    df_combined_chatgpt['success'] = ~np.isnan(df_combined_chatgpt['extracted_revenue'])
    df_combined_chatgpt['ad_hoc_error'] = df_combined_chatgpt.apply(lambda row: calculate_ad_hoc_error(row['revt'], row['extracted_revenue'], row['success']), axis=1)
    df_combined_chatgpt['low'] = df_combined_chatgpt['ad_hoc_error'].apply(bool_low)
    df_combined_chatgpt = df_combined_chatgpt.merge(df_cpi_factors, on='year', how='left')
    df_combined_chatgpt["adj_mcap"] = df_combined_chatgpt["mcap"] + np.log10(df_combined_llama["CPI_factor"])

    same_comp_df_chatgpt = get_same_comp(df_combined_chatgpt, 1980, 2022)

    # Create dummy variables for each year
    year_dummies = pd.get_dummies(same_comp_df_chatgpt['year'], prefix='year', drop_first=True)

    # Combine the year dummies with your independent variable
    X = pd.concat([sm.add_constant(same_comp_df_chatgpt['adj_mcap']), year_dummies], axis=1)

    y = same_comp_df_chatgpt['low']

    # Create and fit the logistic regression model
    model = sm.Logit(y, X.astype(float))
    result = model.fit()

    # Display results
    print(result.summary())

    # Store the result summary as a file
    with open("./regressions_v2/result_summary_same_mcap_gpt4o.txt", "w") as f:
        f.write(str(result.summary()))
