import pandas as pd
import numpy as np

df = pd.read_csv('../data/poc_revenue_mcap_data.csv')

df = df.loc[df['year'] <= 2020]

print(df.columns)

df_funda = pd.read_csv('../data/WRDS/compustat_funda_annual.csv', usecols=['gvkey', 'fyear', 'exchg'] )

df_merged = pd.merge(df, df_funda, on=['gvkey', 'fyear'], how='left')

print(df.shape)

print(df_merged['exchg'].value_counts())