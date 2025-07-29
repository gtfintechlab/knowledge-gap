import pandas as pd
import numpy as np


df_funda = pd.read_csv('../data/WRDS/compustat_funda_annual.csv', usecols=['gvkey', 'fyear', 'exchg', 'fyr'] )

df_funda = df_funda.loc[df_funda['fyear'] >= 1980]

df_funda = df_funda.loc[df_funda['fyear'] <= 2020]

print(df_funda['fyr'].value_counts())

print(df_funda.shape)