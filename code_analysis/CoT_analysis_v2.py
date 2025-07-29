import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *


df_combined_model = pd.read_csv("../data/llm_prompt_outputs_v2/stock_recommendation.csv")

df_combined_model['recommedation'] = df_combined_model['action_response'].apply(extract_recommedation)

# no need to adjust mcap as same year

X = sm.add_constant(df_combined_model['mcap'])

y = df_combined_model['recommedation'].apply(lambda x: (x == "DNK"))


# Create and fit the logistic regression model
model = sm.OLS(y, X)
result = model.fit()

# Display df_combined_model
print(result.summary())


df_combined_model_filter = df_combined_model[df_combined_model["recommedation"] != "SELL"]

X = sm.add_constant(df_combined_model_filter['mcap'])

y = df_combined_model_filter['recommedation'].apply(lambda x: (x == "BUY"))


# Create and fit the logistic regression model
model = sm.OLS(y, X)
result = model.fit()

# Display df_combined_model
print(result.summary())


##############################
df_combined_model_filter = df_combined_model[df_combined_model["recommedation"] != "BUY"]

X = sm.add_constant(df_combined_model_filter['mcap'])

y = df_combined_model_filter['recommedation'].apply(lambda x: (x == "SELL"))


# Create and fit the logistic regression model
model = sm.OLS(y, X)
result = model.fit()

# Display df_combined_model
print(result.summary())