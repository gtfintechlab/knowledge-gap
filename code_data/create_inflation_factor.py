import pandas as pd

df = pd.read_csv("../data/CPIAUCSL_v2.csv")

# Convert the date_column to datetime if it's not already
df['DATE'] = pd.to_datetime(df['DATE'])

# Extract the year from the date_column
df['year'] = df['DATE'].dt.year

df = df.drop_duplicates("year", keep='last')

# value on 2022-12-01 = 298.808
df["CPI_factor"] = 298.808/df["CPIAUCSL"]

df[["year", "CPI_factor"]].to_csv("../data/CPI_factors_v2.csv", index=False)

print(df)