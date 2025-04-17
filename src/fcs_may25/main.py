import marimo as mo

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from icecream import ic
from data_loader import ElectionReconstruction, PREDICTION_FOLDER, IMAGE_PATH
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


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
election_reconstruction.run()

november_turnout = election_reconstruction.november_turnout
election_results = election_reconstruction.election_results
prop_table = election_reconstruction.november_age_pivot
data = election_reconstruction.prepared_election_results
age_range_columns = election_reconstruction.november_proportions_cols
may_prediction_votes = election_reconstruction.may_prediction_votes
prediction_dummies = election_reconstruction.prediction_dummies
prediction_by_age_range = election_reconstruction.prediction_by_age_range
prediction_by_ward = election_reconstruction.prediction_by_ward
prediction_by_precinct = election_reconstruction.prediction_by_precinct
prediction_and_results = election_reconstruction.prediction_and_results
may_current_votes = election_reconstruction.data.current_votes
all_county_voters = election_reconstruction.data.data[['SOS_VOTERID', 'DATE_OF_BIRTH', 'AGE', 'AGE_RANGE', 'WARD', 'PRECINCT_NAME']]
all_county_voters = all_county_voters.set_index('SOS_VOTERID')
may_current_votes = may_current_votes.join(all_county_voters, how='left', on='STATE ID#')
may_current_votes[['WARD', 'PRECINCT_NAME']] = may_current_votes[['WARD', 'PRECINCT_NAME']].astype('object').fillna('Unknown')

may_selected = may_current_votes[['STATE ID#', 'AGE_RANGE', 'WARD', 'PRECINCT_NAME']]

november_age_crosstab = pd.crosstab(
    index=november_turnout['AGE_RANGE'],
    columns=november_turnout['PRECINCT_NAME'],
    dropna=False,
)
november_age_cols = november_age_crosstab.columns.to_list()
november_age_cols.pop(0)

# I want to get the percentage of votes for each precinct broken out by age range
for col in november_age_cols:
    # Create a new column for the percentage
    new_col = f'{col}%'
    # Get total votes for this precinct
    total = november_age_crosstab[col].sum()
    # Calculate percentage for each age range within this precinct
    november_age_crosstab[new_col] = november_age_crosstab[col] / total


may_age_crosstab = pd.crosstab(
    index=may_selected['AGE_RANGE'],
    columns=may_selected['PRECINCT_NAME'],
    dropna=False,
)
may_age_cols = may_age_crosstab.columns.to_list()
may_age_cols.pop(0)

# I want to get the percentage of votes for each precinct broken out by age range
for col in may_age_cols:
    # Create a new column for the percentage
    new_col = f'{col}%'
    # Get total votes for this precinct
    total = may_age_crosstab[col].sum()
    # Calculate percentage for each age range within this precinct
    may_age_crosstab[new_col] = may_age_crosstab[col] / total

november_age_counts = november_turnout['AGE_RANGE'].value_counts().to_frame()
november_age_counts['percent'] = (november_age_counts['count'] / november_age_counts['count'].sum()).round(4)

may_age_counts = may_selected['AGE_RANGE'].value_counts().to_frame()
may_age_counts['percent'] = (may_age_counts['count'] / may_age_counts['count'].sum()).round(4)
# --- Save Results ---
prediction_and_results.to_csv(PREDICTION_FOLDER / 'findlay_results_may6.csv', index=False)
prediction_by_ward.to_csv(PREDICTION_FOLDER / 'findlay_results_may6_by_ward.csv', index=False)
current_counts = prediction_and_results[['prediction_for', 'prediction_against']].sum().rename('votes')
current_counts = current_counts.to_frame().reset_index()
current_counts.columns = ['vote_category', 'votes']
current_counts['pct'] = (current_counts['votes'] / current_counts['votes'].sum()).round(2)
current_counts.to_csv(PREDICTION_FOLDER / 'predicted_results_to_date.csv', index=False)
