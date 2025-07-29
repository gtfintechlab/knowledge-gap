import pandas as pd
import statsmodels.api as sm

# Load the CSV file
data = pd.read_csv("both_leagues_points_pred_results.csv")


# Create dummy variables from the "Year" column
data = pd.get_dummies(data, columns=['Year', 'league'], drop_first=True)

# Define the independent variables (X), which includes Pos, intercept, and Year dummies
X = data[['Pos'] + [col for col in data.columns if col.startswith('Year_') or col.startswith('league_')]]
X = sm.add_constant(X)  # Add the intercept

# Define the dependent variable (Y)
Y = data['success']

# Fit the logistic regression model
model = sm.Logit(Y, X)
result = model.fit()

# Print the summary of the model
print(result.summary())
