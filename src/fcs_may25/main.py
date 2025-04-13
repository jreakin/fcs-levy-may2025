import marimo as mo

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from icecream import ic
from src.fcs_may25.data_loader import ElectionReconstruction, PREDICTION_FOLDER
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

ic.disable()

app = mo.App()

# **Setup**
DOWNLOAD_PATH = Path.home() / 'Downloads'
PROJECT_PATH = Path.home() / 'PyCharmProjects' / 'state-voterfiles'

AGGREGATE_COLS = {
    'VOTED_IN_NOV': 'sum',
    'VOTED_MAY_LEVY': 'sum',
    'NO_IN_NOV_YES_IN_MAY': 'sum',
}

# **Load Election Data**
election_reconstruction = ElectionReconstruction()
election_results = election_reconstruction.data.election_results
election_results['nov_for_pct'] = (election_results['for'] / election_results['total']).round(4)
election_results['nov_against_pct'] = (election_results['against'] / election_results['total']).round(4)
november_turnout = election_reconstruction.data.turnout_data

# --- Precinct-Level Data Preparation ---
# Count voters by precinct and age range
november_counts = november_turnout.groupby(['PRECINCT_NAME', 'AGE_RANGE']).size().reset_index(name='count')
november_total_voters = november_turnout.groupby('PRECINCT_NAME').size().reset_index(name='total')

# Merge and calculate age proportions
merged_counts = november_counts.merge(november_total_voters, on='PRECINCT_NAME')
merged_counts['prop_age'] = merged_counts['count'] / merged_counts['total']

# Pivot to create a proportion table
prop_table = merged_counts.pivot(index='PRECINCT_NAME', columns='AGE_RANGE', values='prop_age').fillna(0).reset_index()

# --- Prepare Election Results ---
election_results = election_results.rename(columns={"precinct": "PRECINCT_NAME"})
data = prop_table.merge(election_results, left_on="PRECINCT_NAME", right_on="PRECINCT_NAME")
data['p_for'] = data['for'] / data['total']

# --- Ecological Regression with Constraints ---
# Select numeric age range columns for X
age_range_columns = [col for col in prop_table.columns if col != 'PRECINCT_NAME']
X = data[age_range_columns].astype(float)
y = data['p_for'].astype(float)



X_np = X.to_numpy()
y_np = y.to_numpy()

# Add age information - modified to work with the actual data structure
age_ranges = ['0-18', '18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+']

# Create a mapping from age range to rank
age_rank_map = {age: i for i, age in enumerate(age_ranges)}

# Add age rank as a new column based on the existing age range columns
# We'll use the column names directly since they match the age ranges
for age_range in age_ranges:
    if age_range in X.columns:
        # Create a temporary column for this age range's rank
        X[f'{age_range}_rank'] = age_rank_map[age_range] * X[age_range]

# Sum up the weighted ranks to get a single age rank column
X['age_rank'] = sum(X[f'{age_range}_rank'] for age_range in age_ranges if f'{age_range}_rank' in X.columns)

# Drop the temporary columns
for age_range in age_ranges:
    if f'{age_range}_rank' in X.columns:
        X = X.drop(columns=[f'{age_range}_rank'])

# Ensure age_rank is numeric
X['age_rank'] = pd.to_numeric(X['age_rank'], errors='coerce').fillna(0)

# Target variable
y = data['p_for']

# Fit constrained least squares model (beta coefficients between 0 and 1)
result = lsq_linear(X_np, y_np, bounds=(0, 1))
beta_k_constrained = result.x

# Map coefficients to age groups
age_groups = sorted(november_turnout['AGE_RANGE'].dropna().astype(str).unique().tolist())
beta_k_constrained_series = pd.Series(beta_k_constrained, index=age_groups)
print("Constrained Beta Coefficients:")
print(beta_k_constrained_series)

# --- OLS Model for Comparison ---
# Fit OLS model to estimate initial group probabilities
model = sm.OLS(y_np, X_np).fit()
beta_k = model.params

# Assess OLS model accuracy
predicted_p_for = model.fittedvalues
actual_p_for = y
r_squared = model.rsquared
ic(f"R-squared: {r_squared:.4f}")
mae = np.mean(np.abs(actual_p_for - predicted_p_for))
ic(f"Mean Absolute Error: {mae:.4f}")
rmse = np.sqrt(np.mean((actual_p_for - predicted_p_for)**2))
ic(f"Root Mean Squared Error: {rmse:.4f}")
predicted_for_votes = predicted_p_for * data['total']
actual_for_votes = data['for']
mae_votes = np.mean(np.abs(actual_for_votes - predicted_for_votes))
ic(f"MAE for 'for' votes: {mae_votes:.2f}")
ic("Beta coefficients:")
ic(beta_k)
if any(beta_k < 0) or any(beta_k > 1):
    ic("Note: Some beta coefficients are outside [0,1]")

# --- Logistic Regression for Individual Vote Predictions ---

# Step 1: Clip beta_k to [0,1] to ensure valid probabilities
beta_k_clipped = np.clip(beta_k, 0, 1)

# Create a dictionary mapping age ranges to their corresponding beta values
age_beta_map = {age: beta for age, beta in zip(age_groups, beta_k_clipped)}

# Step 2: Assign initial probabilities to voters based on age group
november_turnout['prob_for'] = november_turnout['AGE_RANGE'].map(age_beta_map)

# Ensure all probabilities are valid (between 0 and 1) and not NaN
november_turnout['prob_for'] = november_turnout['prob_for'].fillna(0.5)  # Replace NaN with 0.5
november_turnout['prob_for'] = november_turnout['prob_for'].clip(0, 1)  # Clip to [0, 1]

# Step 3: Simulate individual votes based on these probabilities
np.random.seed(42)  # For reproducibility
november_turnout['simulated_vote'] = np.random.binomial(1, november_turnout['prob_for'])

# Step 4: Prepare features for logistic regression (one-hot encode categorical variables)
X_features = pd.get_dummies(november_turnout[['AGE_RANGE', 'PRECINCT_NAME']], drop_first=True)
y_target = november_turnout['simulated_vote']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Step 6: Train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = log_reg.predict(X_test)
ic("\nLogistic Regression Performance:")
ic(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
ic("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Predict probabilities for all voters
november_turnout['predicted_prob_for_logreg'] = log_reg.predict_proba(X_features)[:, 1]

# Step 9: Assign vote predictions based on logistic regression probabilities
november_turnout['vote_prediction_logreg'] = np.where(
    november_turnout['predicted_prob_for_logreg'] < 0.5, 'against',
    np.where(november_turnout['predicted_prob_for_logreg'] > 0.65, 'for', 'swing')
)

# --- Comparison of Predictions ---
# Compare aggregated logistic regression predictions to actual precinct-level p_for
predicted_p_for_logreg = november_turnout.groupby('PRECINCT_NAME')['predicted_prob_for_logreg'].mean()
comparison = pd.DataFrame({
    'actual_p_for': data.set_index('PRECINCT_NAME')['p_for'],
    'predicted_p_for_logreg': predicted_p_for_logreg
})
ic("\nComparison of Aggregated Predictions vs. Actual Precinct Proportions:")
ic(comparison)

# --- Original OLS-Based Predictions ---
# Retain OLS-based predictions for comparison
# Create a dictionary mapping age ranges to their corresponding beta values
beta_k_dict = {age: beta for age, beta in zip(age_groups, beta_k)}
november_turnout['predicted_prob_for'] = november_turnout['AGE_RANGE'].map(beta_k_dict).astype(float)
november_turnout['vote_prediction'] = np.where(
    november_turnout['predicted_prob_for'] < 0.45, 'against',
    np.where(november_turnout['predicted_prob_for'] >= 0.55, 'for', 'swing')
)
prediction_dummies = pd.get_dummies(november_turnout['vote_prediction'], prefix='prediction')
november_turnout[list(prediction_dummies.columns)] = prediction_dummies

prediction_by_precinct = november_turnout.groupby(['ward', 'PRECINCT_NAME'])[list(prediction_dummies.columns)].sum().reset_index()
results_and_prediction = election_results.merge(prediction_by_precinct, left_on='PRECINCT_NAME', right_on='PRECINCT_NAME')
results_and_prediction['for_diff'] = results_and_prediction['for'] - results_and_prediction['prediction_for']
results_and_prediction['against_diff'] = results_and_prediction['against'] - results_and_prediction['prediction_against']
result_agg_cols = {x: 'sum' for x in list(prediction_dummies.columns)}
result_agg_cols['SOS_VOTERID'] = 'count'
may_prediction_votes = november_turnout[november_turnout['VOTED_MAY_LEVY'] == 1].groupby(['ward', 'PRECINCT_NAME', 'AGE_RANGE']).agg(
    result_agg_cols
).rename(columns={'SOS_VOTERID': 'total_votes'})
prediction_by_precinct = may_prediction_votes.groupby('PRECINCT_NAME')[['total_votes'] + list(prediction_dummies.columns)].sum().reset_index()
prediction_by_precinct['pct_for'] = (prediction_by_precinct['prediction_for'] / prediction_by_precinct['total_votes']).round(2)
prediction_by_precinct['pct_against'] = (prediction_by_precinct['prediction_against'] / prediction_by_precinct['total_votes']).round(2)
prediction_by_precinct['pct_swing'] = (prediction_by_precinct['prediction_swing'] / prediction_by_precinct['total_votes']).round(2)

prediction_and_results = prediction_by_precinct.merge(election_results, left_on='PRECINCT_NAME', right_on='PRECINCT_NAME')
prediction_and_results['better_than_nov'] = prediction_and_results['pct_for'] > prediction_and_results['nov_for_pct']
prediction_and_results['winning'] = prediction_and_results['prediction_for'] >= prediction_and_results['prediction_against']

prediction_by_ward = may_prediction_votes.groupby('ward')[['total_votes'] + list(prediction_dummies.columns)].sum().reset_index()
prediction_by_ward['pct_for'] = (prediction_by_ward['prediction_for'] / prediction_by_ward['total_votes']).round(2)
prediction_by_ward['pct_against'] = (prediction_by_ward['prediction_against'] / prediction_by_ward['total_votes']).round(2)
prediction_by_ward['pct_swing'] = (prediction_by_ward['prediction_swing'] / prediction_by_ward['total_votes']).round(2)
prediction_by_ward['ward_pct_of_vote'] = (prediction_by_ward['total_votes'] / prediction_by_ward['total_votes'].sum()).round(2)

prediction_by_age_range = may_prediction_votes.groupby('AGE_RANGE')[list(prediction_dummies.columns)].sum().reset_index()
prediction_by_age_range['total_votes'] = prediction_by_age_range['prediction_for'] + prediction_by_age_range['prediction_against'] + prediction_by_age_range['prediction_swing']
prediction_by_age_range['pct_for'] = (prediction_by_age_range['prediction_for'] / prediction_by_age_range['total_votes']).round(2)
prediction_by_age_range['pct_against'] = (prediction_by_age_range['prediction_against'] / prediction_by_age_range['total_votes']).round(2)

prediction_and_results.to_csv(PREDICTION_FOLDER / 'findlay_results_may6.csv', index=False)
prediction_by_ward.to_csv(PREDICTION_FOLDER / 'findlay_results_may6_by_ward.csv', index=False)
current_counts = prediction_and_results[['prediction_for', 'prediction_against']].sum().rename('votes')
current_counts = current_counts.to_frame().reset_index()
current_counts.columns = ['vote_category', 'votes']
current_counts['pct'] = (current_counts['votes'] / current_counts['votes'].sum()).round(2)
current_counts.to_csv(PREDICTION_FOLDER / 'predicted_results_to_date.csv', index=False)

# ward_data = election_reconstruction.data.turnout_data
# ward_groupby = ward_data.groupby('ward').agg(AGGREGATE_COLS).reset_index()
# ward_groupby['pct_of_vote_nov'] = (ward_groupby['VOTED_IN_NOV'] / ward_groupby['VOTED_IN_NOV'].sum()).round(2)
# ward_groupby['pct_of_vote_may'] = (ward_groupby['VOTED_MAY_LEVY'] / ward_groupby['VOTED_MAY_LEVY'].sum()).round(2)
# precinct_groupby = ward_data.groupby(['PRECINCT_NAME', 'AGE_RANGE']).agg(AGGREGATE_COLS).reset_index()
#
# for precinct in precinct_groupby['PRECINCT_NAME']:
#     precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'pct_of_vote_nov'] = (
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'VOTED_IN_NOV'] /
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'VOTED_IN_NOV'].sum()
#     ).round(2)
#     precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'pct_of_vote_may'] = (
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'VOTED_MAY_LEVY'] /
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'VOTED_MAY_LEVY'].sum()
#     ).fillna(0).round(2)
#     precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'MAY_HIGHER_THAN_NOV'] = (
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'pct_of_vote_may'] >
#         precinct_groupby.loc[precinct_groupby['PRECINCT_NAME'] == precinct, 'pct_of_vote_nov']
#     )
#
#
# election_groupby = election_results.groupby('ward').agg(
#     {
#     'for': 'sum',
#     'against': 'sum',
#     'total': 'sum'
#     }
# ).reset_index()
# election_groupby['diff'] = (election_groupby['for'] - election_groupby['against'])
# merge_ward_and_results = ward_groupby.merge(
#     election_groupby,
#     how='inner',
#     left_on='ward',
#     right_on='ward'
# )
# merge_ward_and_results['for_percent'] = (merge_ward_and_results['for'] / merge_ward_and_results['total']).round(2)
# merge_ward_and_results['against_percent'] = (merge_ward_and_results['against'] / merge_ward_and_results['total']).round(2)
# merge_ward_and_results['NOV_VOTED_YES'] = merge_ward_and_results['for_percent'] >= merge_ward_and_results['against_percent']
# merge_ward_and_results['MAY_HIGHER_THAN_NOV'] = (
#     merge_ward_and_results['pct_of_vote_may'] > merge_ward_and_results['pct_of_vote_nov']
# )

# ward_ct = pd.crosstab(
# index=ward_data['ward'],
# columns=ward_data['CATEGORY'],
# ).reset_index()
# category_cols = ward_ct.columns[1:]
# merge_ward_and_results = merge_ward_and_results.merge(
#     ward_ct,
#     how='inner',
#     left_on='ward',
#     right_on='ward'
# )
# category_pct_cols = []
# for col in category_cols:
#     category_pct_cols.append(_fmt_col := f"{col}-%")
#     merge_ward_and_results[_fmt_col] = (merge_ward_and_results[col] / merge_ward_and_results['total']).round(2)
#
# # Add ward information to precinct_groupby
# precinct_groupby['ward'] = precinct_groupby['PRECINCT_NAME'].str[:-1]