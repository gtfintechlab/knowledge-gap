import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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

df = df.loc[(df['year'] < 2023)]

df_cpi_factors = pd.read_csv("../data/CPI_factors_v2.csv")

df = df.merge(df_cpi_factors, on='year', how='left')
df["adj_mcap"] = df["mcap"] + np.log10(df["CPI_factor"])

# plt.hist(df["adj_mcap"], bins=20)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Histogram')
# plt.show()

# 5 brackets <8.00, 8.xx, 9.xx, >=10.00
# Apply categorization
df['adj_mcap_category'] = df['adj_mcap'].apply(categorize_mcap)

# Sampling
np.random.seed(1729)

n = 50 # Desired maximum samples per group

# Define a custom sampling function
def custom_sample(group):
    num_samples = min(len(group), n)  # Determine the number of samples to take from the group
    return group.sample(n=num_samples, replace=False)  # Sample without replacement


# sampled_df = df.groupby(['year', 'adj_mcap_category']).sample(n=50, replace=False)
sampled_df = df.groupby(['year', 'adj_mcap_category']).apply(custom_sample).reset_index(drop=True)

print(sampled_df.shape)

# sampled_df.to_csv("../data/sampled_data_v2.csv", index=False)