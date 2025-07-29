import pandas as pd
import numpy as np


######################### Load and extract relevant data from link file #########################

df_link = pd.read_sas('../data/WRDS/link.sas7bdat', format = 'sas7bdat', encoding="utf-8")
# print(df_link.columns)

df_link = df_link[['GVKEY', 'LPERMNO', 'TIC', 'CONM', 'SIC', 'CIK', 'LINKDT', 'LINKENDDT']]

df_link = df_link.dropna(subset=['GVKEY', 'LPERMNO', 'TIC', 'CONM', 'SIC', 'CIK'])

df_link = df_link.drop_duplicates(subset=['GVKEY', 'LPERMNO'], keep='last')

df_link['CIK'] = df_link['CIK'].astype(int)
df_link['GVKEY'] = df_link['GVKEY'].astype(int)
df_link['LPERMNO'] = df_link['LPERMNO'].astype(int)

# print(df_link.head())




######################### Load and extract relevant data from CRSP-MSF file #########################

df_msf = pd.read_csv('../data/WRDS/msf_full_v2.csv', usecols=['PERMNO', 'date', 'SHRCD', 'PRC', 'SHROUT']) 
df_msf = df_msf.dropna()

# keep only SHRCD=10 and SHRCD=11
df_msf = df_msf.loc[(df_msf['SHRCD'] == 10.0) | (df_msf['SHRCD'] == 11.0)]

# fix negative price problem
df_msf['PRC'] = df_msf['PRC'].abs()

# get market cap. SHROUT is the number of publicly held shares, recorded in thousands
df_msf['mcap'] = np.log10(df_msf['PRC']*df_msf['SHROUT']*1000)

# get year 
df_msf['year'] = pd.DatetimeIndex(df_msf['date']).year

# keep last observation per company-year
df_msf = df_msf.drop_duplicates(subset=['PERMNO', 'year'], keep='last')

print(df_msf.shape)
# print(df_msf.head())







######################### Load and extract relevant data from Compustat Annual Fundamental data file #########################

# note: revt numbers are in millions

df_funda = pd.read_csv('../data/WRDS/compustat_funda_annual_v2.csv', usecols=['gvkey', 'fyear', 'revt'] )
df_funda = df_funda.dropna()

df_funda = df_funda.drop_duplicates(subset=['gvkey', 'fyear'], keep='last')

print(df_funda.shape)
# print(df_funda.head())
print("Number of unique companies Funda dataset: ", df_funda['gvkey'].nunique())


######################### Merge datasets and save file #########################
df_merged = pd.merge(df_msf, df_link, how='left', left_on='PERMNO', right_on='LPERMNO')
print(df_merged.shape)
df_merged = pd.merge(df_merged, df_funda, how='left', left_on=['GVKEY', 'year'], right_on=['gvkey', 'fyear'])

print(df_merged.shape)

df_merged = df_merged.dropna(subset=['PERMNO', 'GVKEY', 'year', 'CONM', 'mcap', 'revt'])

print(df_merged.shape)
# print(df_merged.head())
df_merged = df_merged.drop_duplicates(subset=['GVKEY', 'year'])
print(df_merged.shape)

df_merged = df_merged.loc[df_merged['year'] >= 1980]
print(df_merged.shape)


df_merged = df_merged.loc[df_merged['year'] <= 2024]
print("Number of unique companies in our full dataset: ", df_merged['gvkey'].nunique())

df_merged.to_csv('../data/poc_revenue_mcap_data_v2.csv', index=False)