import marimo as mo

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from icecream import ic
from sklearn.compose import ColumnTransformer
from data_loader import ElectionReconstruction, PREDICTION_FOLDER, IMAGE_PATH
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from category_encoders import BinaryEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

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

data = election_reconstruction.data.data
voters_by_precinct = data

precinct_list_data = []
for precinct in voters_by_precinct['PRECINCT_NAME'].unique():
    precinct_data = voters_by_precinct[voters_by_precinct['PRECINCT_NAME'] == precinct]
    precinct_group = precinct_data.groupby(['WARD', 'PRECINCT_NAME', 'AGE_RANGE']).agg(AGGREGATE_COLS).reset_index()
    group_cols = list(precinct_group.columns)
    group_cols.pop(0)
    group_cols.pop(0)
    group_cols.pop(0)
    for col in group_cols:
        precinct_group[col] = precinct_group[col].astype('int')
        precinct_group[f"{col}_pct"] = precinct_group[col] / precinct_group[col].sum()
    precinct_group[group_cols] = precinct_group[group_cols]
    precinct_list_data.append(precinct_group)

precinct_percent_data = pd.concat(precinct_list_data)
precinct_crosstab = pd.crosstab(
    index=[precinct_percent_data['WARD'], precinct_percent_data['PRECINCT_NAME']],
    columns=precinct_percent_data['AGE_RANGE'],
    values=precinct_percent_data['VOTED_IN_NOV'],
    aggfunc='sum',
).reset_index()
precinct_crosstab_cols = precinct_crosstab.columns.to_list()
precinct_crosstab[precinct_crosstab_cols[2:]] = precinct_crosstab[precinct_crosstab_cols[2:]].astype('int')
precinct_crosstab['total'] = precinct_crosstab[precinct_crosstab_cols[2:]].sum(axis=1)
precinct_crosstab = precinct_crosstab[precinct_crosstab['total'] > 0]

results = election_reconstruction.data.election_results
results_change_index = results.copy().set_index('precinct')
precinct_ct_plus_results = precinct_crosstab.join(results_change_index, how='left', on='PRECINCT_NAME', rsuffix='_results', lsuffix='_count')

# November Voters Pre-Cleaning
november_voters = voters_by_precinct[['PRECINCT_NAME', 'AGE_RANGE', 'PARTY_AFFILIATION', 'WARD', 'VOTED_IN_NOV', 'VOTED_MAY_LEVY', 'NO_IN_NOV_YES_IN_MAY']].copy()
november_voters['AGE_RANGE_CAT'] = pd.Categorical(
    november_voters['AGE_RANGE'],
    categories=sorted(november_voters['AGE_RANGE'].unique()),
    ordered=True
)
november_voters['WARD_CAT'] = pd.Categorical(
    november_voters['WARD'],
    categories=sorted(november_voters['WARD'].unique()),
    ordered=False
)
november_voters['PRECINCT_NAME_CAT'] = pd.Categorical(
    november_voters['PRECINCT_NAME'],
    categories=sorted(november_voters['PRECINCT_NAME'].unique()),
    ordered=False
)

november_voters['PARTY_AFFILIATION_CAT'] = pd.Categorical(
    november_voters['PARTY_AFFILIATION'],
    categories=sorted(november_voters['PARTY_AFFILIATION'].unique()),
    ordered=False
)

# First do all the stats calculations and merges, then create the interaction features

# For November stats
nov_ward_stats = november_voters.groupby('WARD').agg({
    'VOTED_IN_NOV': ['mean', 'sum']
}).reset_index()
nov_ward_stats.columns = ['WARD', 'nov_ward_turnout_rate', 'nov_ward_voter_count']

# For May stats
may_ward_stats = november_voters.groupby('WARD').agg({
    'VOTED_MAY_LEVY': ['mean', 'sum']
}).reset_index()
may_ward_stats.columns = ['WARD', 'may_ward_turnout_rate', 'may_ward_voter_count']

# For precinct-level stats
nov_precinct_stats = november_voters.groupby('PRECINCT_NAME').agg({
    'VOTED_IN_NOV': ['mean', 'sum']
}).reset_index()
nov_precinct_stats.columns = ['PRECINCT_NAME', 'nov_precinct_turnout_rate', 'nov_precinct_voter_count']

may_precinct_stats = november_voters.groupby('PRECINCT_NAME').agg({
    'VOTED_MAY_LEVY': ['mean', 'sum']
}).reset_index()
may_precinct_stats.columns = ['PRECINCT_NAME', 'may_precinct_turnout_rate', 'may_precinct_voter_count']

# Do all the merges
november_voters = november_voters.merge(nov_precinct_stats, how='left', on='PRECINCT_NAME')
november_voters = november_voters.merge(may_precinct_stats, how='left', on='PRECINCT_NAME')
november_voters = november_voters.merge(nov_ward_stats, how='left', on='WARD')
november_voters = november_voters.merge(may_ward_stats, how='left', on='WARD')

# Now create the turnout difference features
november_voters['turnout_diff'] = november_voters['nov_ward_turnout_rate'] - november_voters['may_ward_turnout_rate']
november_voters['precinct_turnout_diff'] = november_voters['nov_precinct_turnout_rate'] - november_voters['may_precinct_turnout_rate']

# First create the interaction features as strings
november_voters['AGE_PARTY'] = (
    november_voters['AGE_RANGE_CAT'].astype(str).str.strip() + 
    '_' + 
    november_voters['PARTY_AFFILIATION_CAT'].astype(str).str.strip()
)
november_voters['WARD_PARTY'] = (
    november_voters['WARD_CAT'].astype(str).str.strip() + 
    '_' + 
    november_voters['PARTY_AFFILIATION_CAT'].astype(str).str.strip()
)
november_voters['PRECINCT_PARTY_TURNOUT'] = (
    november_voters['PRECINCT_NAME_CAT'].astype(str) + '_' +
    november_voters['PARTY_AFFILIATION_CAT'].astype(str)
)

# Convert all nominal features to categorical type
for col in ['AGE_RANGE_CAT', 'WARD_CAT', 'PRECINCT_NAME_CAT', 'PARTY_AFFILIATION_CAT']:
    november_voters[col] = november_voters[col].cat.add_categories('Unknown').fillna('Unknown')

for col in ['AGE_PARTY', 'WARD_PARTY', 'PRECINCT_PARTY_TURNOUT']:
    # For interaction features, create as new categorical
    november_voters[col] = pd.Categorical(
        november_voters[col],
        categories=sorted(november_voters[col].unique()),
        ordered=False
    )
    # Now we can safely add Unknown category and fill NAs
    november_voters[col] = november_voters[col].cat.add_categories('Unknown').fillna('Unknown')

november_voters['VOTED_IN_NOV'] = november_voters['VOTED_IN_NOV'].astype('int')
november_voters['VOTED_MAY_LEVY'] = november_voters['VOTED_MAY_LEVY'].astype('int')
november_voters['NO_IN_NOV_YES_IN_MAY'] = november_voters['NO_IN_NOV_YES_IN_MAY'].astype('int')
november_voters = november_voters.merge(results_change_index, left_on='PRECINCT_NAME', right_on='precinct', suffixes=('_results', '_results_stats'))
november_voters = november_voters.rename(columns={'for': 'november_for', 'against': 'november_against', 'total': 'november_total'})

ordinal_features = ['AGE_RANGE_CAT']
nominal_features = ['WARD_CAT', 'PRECINCT_NAME_CAT', 'PARTY_AFFILIATION_CAT', 'AGE_PARTY', 'WARD_PARTY', 'PRECINCT_PARTY_TURNOUT']
numeric_features = ['turnout_diff', 'precinct_turnout_diff', 'nov_ward_turnout_rate', 'may_ward_turnout_rate', 'nov_for_pct', 'nov_against_pct']

precinct_patterns = november_voters.groupby('PRECINCT_NAME').agg({
    'VOTED_IN_NOV': ['mean', 'std'],  # Historical turnout patterns
    'VOTED_MAY_LEVY': ['mean', 'std']  # May levy patterns
}).reset_index()

precinct_history = november_voters.groupby('PRECINCT_NAME').agg({
    'VOTED_IN_NOV': 'mean',
    'VOTED_MAY_LEVY': 'mean'
}).reset_index()

november_voters = november_voters.merge(
    precinct_history,
    on='PRECINCT_NAME',
    suffixes=('', '_precinct_avg')
)
numeric_features.extend(['VOTED_IN_NOV_precinct_avg', 'VOTED_MAY_LEVY_precinct_avg'])

# Add more voter history features
november_voters['voted_both'] = ((november_voters['VOTED_IN_NOV'] == 1) & 
                                (november_voters['VOTED_MAY_LEVY'] == 1)).astype(int)
november_voters['skipped_both'] = ((november_voters['VOTED_IN_NOV'] == 0) & 
                                  (november_voters['VOTED_MAY_LEVY'] == 0)).astype(int)

# Add age-based turnout rates
age_turnout = november_voters.groupby('AGE_RANGE_CAT')['VOTED_IN_NOV'].mean().reset_index()
age_turnout.columns = ['AGE_RANGE_CAT', 'age_group_turnout']
november_voters = november_voters.merge(age_turnout, on='AGE_RANGE_CAT')

# Add party-based turnout rates
party_turnout = november_voters.groupby('PARTY_AFFILIATION_CAT')['VOTED_IN_NOV'].mean().reset_index()
party_turnout.columns = ['PARTY_AFFILIATION_CAT', 'party_turnout']
november_voters = november_voters.merge(party_turnout, on='PARTY_AFFILIATION_CAT')

# Update numeric features
numeric_features.extend([
    'voted_both', 'skipped_both', 
    'age_group_turnout', 'party_turnout'
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('ord', OrdinalEncoder(), ordinal_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), nominal_features)
    ]
)

# Create base models with their own parameters
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

svc = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', lr),
        ('svc', svc)
    ],
    voting='soft'
)

# Create pipeline with voting classifier
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ]
)

# Update parameter grid for the ensemble
param_grid = {
    'classifier__rf__n_estimators': [200, 300, 400],
    'classifier__rf__max_depth': [15, 20, 25],
    'classifier__rf__min_samples_leaf': [10, 15, 20],
    'classifier__lr__C': [0.1, 1.0, 10.0],
    'classifier__svc__C': [0.1, 1.0, 10.0],
    'classifier__svc__kernel': ['rbf', 'linear']
}

# Create grid search with multiple metrics
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model
X = november_voters[ordinal_features + nominal_features + numeric_features]
y = november_voters['VOTED_IN_NOV']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best model and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Get feature names from the fitted preprocessor
feature_names = (
    preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features).tolist() +
    preprocessor.named_transformers_['ord'].get_feature_names_out(ordinal_features).tolist() +
    preprocessor.named_transformers_['cat'].get_feature_names_out(nominal_features).tolist()
)

# Get feature importances from the Random Forest classifier in the ensemble
rf_classifier = best_model.named_steps['classifier'].named_estimators_['rf']
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_classifier.feature_importances_
})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Print top 10 most important features
print("\nTop 10 most important features:")
print(feature_importances.head(10))

# Visualize feature importances
plt.figure(figsize=(12, 6))
feature_importances.head(15).plot(x='feature', y='importance', kind='bar')
plt.title('Top 15 Most Important Features (from Random Forest)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print model performance metrics
print("\nModel Performance Summary:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# election_reconstruction.run()

# november_turnout = election_reconstruction.november_turnout
# election_results = election_reconstruction.election_results
# prop_table = election_reconstruction.november_age_pivot
# data = election_reconstruction.prepared_election_results
# age_range_columns = election_reconstruction.november_proportions_cols
# may_prediction_votes = election_reconstruction.may_prediction_votes
# prediction_dummies = election_reconstruction.prediction_dummies
# prediction_by_age_range = election_reconstruction.prediction_by_age_range
# prediction_by_ward = election_reconstruction.prediction_by_ward
# prediction_by_precinct = election_reconstruction.prediction_by_precinct
# prediction_and_results = election_reconstruction.prediction_and_results
# may_current_votes = election_reconstruction.data.current_votes
# all_county_voters = election_reconstruction.data.data[['SOS_VOTERID', 'DATE_OF_BIRTH', 'AGE', 'AGE_RANGE', 'WARD', 'PRECINCT_NAME']]
# all_county_voters = all_county_voters.set_index('SOS_VOTERID')
# may_current_votes = may_current_votes.join(all_county_voters, how='left', on='STATE ID#')
# may_current_votes[['WARD', 'PRECINCT_NAME']] = may_current_votes[['WARD', 'PRECINCT_NAME']].astype('object').fillna('Unknown')

# may_selected = may_current_votes[['STATE ID#', 'AGE_RANGE', 'WARD', 'PRECINCT_NAME']]

# november_age_crosstab = pd.crosstab(
#     index=november_turnout['AGE_RANGE'],
#     columns=november_turnout['PRECINCT_NAME'],
#     dropna=False,
# )
# november_age_cols = november_age_crosstab.columns.to_list()
# november_age_cols.pop(0)

# # I want to get the percentage of votes for each precinct broken out by age range
# for col in november_age_cols:
#     # Create a new column for the percentage
#     new_col = f'{col}%'
#     # Get total votes for this precinct
#     total = november_age_crosstab[col].sum()
#     # Calculate percentage for each age range within this precinct
#     november_age_crosstab[new_col] = november_age_crosstab[col] / total


# may_age_crosstab = pd.crosstab(
#     index=may_selected['AGE_RANGE'],
#     columns=may_selected['PRECINCT_NAME'],
#     dropna=False,
# )
# may_age_cols = may_age_crosstab.columns.to_list()
# may_age_cols.pop(0)

# # I want to get the percentage of votes for each precinct broken out by age range
# for col in may_age_cols:
#     # Create a new column for the percentage
#     new_col = f'{col}%'
#     # Get total votes for this precinct
#     total = may_age_crosstab[col].sum()
#     # Calculate percentage for each age range within this precinct
#     may_age_crosstab[new_col] = may_age_crosstab[col] / total

# november_age_counts = november_turnout['AGE_RANGE'].value_counts().to_frame()
# november_age_counts['percent'] = (november_age_counts['count'] / november_age_counts['count'].sum()).round(4)
# november_age_counts = november_age_counts.sort_values(by='AGE_RANGE', ascending=True)

# may_age_counts = may_selected['AGE_RANGE'].value_counts().to_frame()
# may_age_counts['percent'] = (may_age_counts['count'] / may_age_counts['count'].sum()).round(4)
# may_age_counts = may_age_counts.sort_values(by='AGE_RANGE', ascending=True)
# # --- Save Results ---
# prediction_and_results.to_csv(PREDICTION_FOLDER / 'findlay_results_may6.csv', index=False)
# prediction_by_ward.to_csv(PREDICTION_FOLDER / 'findlay_results_may6_by_ward.csv', index=False)
# current_counts = prediction_and_results[['prediction_for', 'prediction_against']].sum().rename('votes')
# current_counts = current_counts.to_frame().reset_index()
# current_counts.columns = ['vote_category', 'votes']
# current_counts['pct'] = (current_counts['votes'] / current_counts['votes'].sum()).round(2)
# current_counts.to_csv(PREDICTION_FOLDER / 'predicted_results_to_date.csv', index=False)
