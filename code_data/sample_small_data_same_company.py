import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# carries survival bias in the data sampled

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


df = pd.read_csv('../data/poc_revenue_mcap_data_v2.csv')

# only keep starting 2013
df = df.loc[(df['year'] >= 2013) & (df['year'] < 2025)]

df_cpi_factors = pd.read_csv("../data/CPI_factors_v2.csv")

df = df.merge(df_cpi_factors, on='year', how='left')
df["adj_mcap"] = df["mcap"] + np.log10(df["CPI_factor"])

# 5 brackets <8.00, 8.xx, 9.xx, >=10.00
# Apply categorization
df['adj_mcap_category'] = df['adj_mcap'].apply(categorize_mcap)



########### keep the same company over time 
# Define the year range
year_range = set(range(2013, 2025))

# Identify GVKEYs that have all years in the range
valid_gvkeys = df.groupby('GVKEY').filter(lambda x: year_range.issubset(set(x['year']))).GVKEY.unique()

# Filter the original DataFrame to keep rows with those GVKEYs
filtered_df = df[df['GVKEY'].isin(valid_gvkeys)]


# Sampling
np.random.seed(1729)

n = 50 # Desired maximum samples per group

# Filter for year 2023
df_2023 = filtered_df[filtered_df['year'] == 2023]

# Sample 50 GVKEYs from each category of label
sampled_gvkeys = df_2023.groupby('adj_mcap_category').apply(lambda x: x['GVKEY'].sample(n=min(len(x), 50), replace=False)).reset_index(drop=True)

# Use the sampled GVKEYs to filter the original DataFrame
sampled_df = filtered_df[filtered_df['GVKEY'].isin(sampled_gvkeys)]


# Count the number of rows for each year
firm_counts = sampled_df.groupby(['GVKEY']).size()

print("Number of firms: ", firm_counts)

print(sampled_df.shape)

sampled_df.to_csv("../data/sampled_data_same_over_time_v2.csv", index=False)