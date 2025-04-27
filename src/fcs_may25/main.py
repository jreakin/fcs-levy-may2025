# import marimo as mo
from category_encoders import TargetEncoder
from icecream import ic
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from data_loader import FindlayVoterFile, FindlayVoterFileConfig as vf_config, NovemberResultsColumns as nov_results
from config import FindlayLinearModelFeatureLists as ml_cat
from prediction import (
    FindlayPredictionModel,
    test_max_iter_impact,
    plot_regression_results,
    plot_feature_importance
)
from decision_tree_ import FindlayDecisionTree
from monte_carlo import MonteCarloVoterSimulation, MonteCarloConfig

print("Loading data and initializing models...")
category_data = ml_cat()
voterfile = FindlayVoterFile()

election_results = voterfile.election_results
election_results_against_mean = election_results['nov_against_share'].mean()
election_results_for_mean = election_results['nov_for_share'].mean()
election_results_against_std = election_results['nov_against_share'].std()
thresholds = election_results['nov_for_share'].quantile([0.15, 0.4, 0.6, 0.85])

election_results['nov_for_share'].hist()
plt.show()

m_data = voterfile.model_data

age_ward_city = pd.crosstab(m_data['AGE_RANGE'], m_data['WARD'], margins=True, normalize='all')
age_precinct_city = pd.crosstab(m_data['AGE_RANGE'], m_data['PRECINCT_NAME'], margins=True, normalize='all')

# 2. Get the age distribution within each ward/precinct
age_ward_within = pd.crosstab(m_data['AGE_RANGE'], m_data['WARD'], margins=True, normalize='columns')
age_precinct_within = pd.crosstab(m_data['AGE_RANGE'], m_data['PRECINCT_NAME'], margins=True, normalize='columns')

# First calculate primary turnouts
primary_elections = list(vf_config.PRIMARY_COLUMNS.keys())
general_elections = list(vf_config.GENERAL_COLUMNS.keys())
m_data[primary_elections] = m_data[primary_elections].astype(bool)
m_data[general_elections] = m_data[general_elections].astype(bool)

def calculate_turnout(data: pd.DataFrame, elections: list[str], by_column: str, district_level: str):

    turnout_list = []
    election_type = elections[0].split('-')[0].lower()
    ic(f"Calculating {election_type} turnout for {by_column} in {district_level}")
    _district_level_name = district_level if district_level != 'PRECINCT_NAME' else 'precinct'
    _by_column_name = by_column if by_column != 'PARTY_AFFILIATION' else 'party'
    _district_level_and_column_name = f'{_district_level_name}_{_by_column_name}'.lower()
    for election in elections:
        # Get both raw counts and percentages
        raw_counts = pd.crosstab(
            [data[by_column], data[district_level]], 
            data[election]
        )
        
        normalized = pd.crosstab(
            [data[by_column], data[district_level]], 
            data[election],
            normalize='index'
        )

        if True in raw_counts.columns:
            counts = raw_counts[True]
            rates = normalized[True]
        elif 1 in raw_counts.columns:
            counts = raw_counts[1]
            rates = normalized[1]
        
        # Reset index to convert to table format
        counts = counts.reset_index()
        rates = rates.reset_index()
        
        # Rename columns for clarity
        counts.columns = [by_column, district_level, f'{election}_count']
        rates.columns = [by_column, district_level, f'{election}_rate']
        
        # Merge counts and rates
        turnout = counts.merge(rates, on=[by_column, district_level])
        turnout_list.append(turnout)

    # Merge all turnouts into one table
    turnout_table = turnout_list[0]
    for df in turnout_list[1:]:
        turnout_table = turnout_table.merge(
            df[[by_column, district_level] + [col for col in df.columns if col not in [by_column, district_level]]], 
            on=[by_column, district_level]
        )
    
    # Add average turnout across all elections
    rate_cols = [col for col in turnout_table.columns if col.endswith('_rate')]
    count_cols = [col for col in turnout_table.columns if col.endswith('_count')]
    
    # Calculate averages
    turnout_table[_rate_col_name := f'{election_type}_avg_{_district_level_and_column_name}_rate'.lower()] = turnout_table[rate_cols].mean(axis=1)
    turnout_table[_count_col_name := f'{election_type}_avg_{_district_level_and_column_name}_count'.lower()] = turnout_table[count_cols].mean(axis=1)
    
    # Calculate relative participation (how much this group participates compared to average)
    avg_participation = turnout_table[_rate_mean_col_name := f'{election_type}_avg_{_district_level_and_column_name}_rate'.lower()].mean()
    turnout_table[_participation_col_name := f'{election_type}_{_district_level_and_column_name}_relative_participation'.lower()] = (
        turnout_table[_rate_col_name] / avg_participation
    )
    turnout_table = turnout_table[[
        by_column, 
        district_level,
        _rate_col_name, 
        _count_col_name, 
        _participation_col_name
    ]].rename(columns={
        _rate_col_name : (_new_rate_col_name := f'{election_type}_avg_{_district_level_and_column_name}_by_{_by_column_name}_rate'.lower()),
        _count_col_name : (_new_count_col_name := f'{election_type}_avg_{_district_level_and_column_name}_by_{_by_column_name}_count'.lower()),
        _participation_col_name : (_new_participation_col_name := f'{election_type}_{_district_level_and_column_name}_by_{_by_column_name}_relative_participation'.lower())
    })
    _current_features = [_new_rate_col_name, _new_count_col_name, _new_participation_col_name]
    ml_cat.numerical_features.extend(_current_features)
    return turnout_table

m_data = m_data.merge(calculate_turnout(m_data, primary_elections, by_column='AGE_RANGE', district_level='WARD'), on=['AGE_RANGE', 'WARD'], how='left')
m_data = m_data.merge(calculate_turnout(m_data, general_elections, by_column='AGE_RANGE', district_level='WARD'), on=['AGE_RANGE', 'WARD'], how='left')

m_data = m_data.merge(calculate_turnout(m_data, primary_elections, by_column='AGE_RANGE', district_level='PRECINCT_NAME'), on=['AGE_RANGE', 'PRECINCT_NAME'], how='left')
m_data = m_data.merge(calculate_turnout(m_data, general_elections, by_column='AGE_RANGE', district_level='PRECINCT_NAME'), on=['AGE_RANGE', 'PRECINCT_NAME'], how='left')

m_data = m_data.merge(calculate_turnout(m_data, primary_elections, by_column='PARTY_AFFILIATION', district_level='WARD'), on=['PARTY_AFFILIATION', 'WARD'], how='left')
m_data = m_data.merge(calculate_turnout(m_data, general_elections, by_column='PARTY_AFFILIATION', district_level='WARD'), on=['PARTY_AFFILIATION', 'WARD'], how='left')

m_data = m_data.merge(calculate_turnout(m_data, primary_elections, by_column='PARTY_AFFILIATION', district_level='PRECINCT_NAME'), on=['PARTY_AFFILIATION', 'PRECINCT_NAME'], how='left')
m_data = m_data.merge(calculate_turnout(m_data, general_elections, by_column='PARTY_AFFILIATION', district_level='PRECINCT_NAME'), on=['PARTY_AFFILIATION', 'PRECINCT_NAME'], how='left')

# Category Features
m_data[age_range_cat := 'AGE_RANGE_CAT'] = pd.Categorical(m_data['AGE_RANGE'], categories=sorted(m_data['AGE_RANGE'].unique()), ordered=True)
m_data[party_cat := 'PARTY_CAT'] = pd.Categorical(m_data['PARTY_AFFILIATION'], categories=['D', 'I', 'R'], ordered=True)
ml_cat.category_features.extend([age_range_cat, party_cat])


# Interaction Features
m_data[age_ward := 'AGE_WARD'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['WARD'].astype(str)
m_data[age_precinct := 'AGE_PRECINCT'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['PRECINCT_NAME'].astype(str)
m_data[age_party := 'AGE_PARTY'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['PARTY_AFFILIATION'].astype(str)
m_data[p_score_last4_cat := 'P_SCORE_LAST4_CAT'] = pd.cut(
    m_data['P_SCORE'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[g_score_last4_cat := 'G_SCORE_LAST4_CAT'] = pd.cut(
    m_data['G_SCORE'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[p_score_all_cat := 'P_SCORE_ALL_CAT'] = pd.cut(
    m_data['P_SCORE_ALL'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[g_score_all_cat := 'G_SCORE_ALL_CAT'] = pd.cut(
    m_data['G_SCORE_ALL'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[p_score_last4_ward := 'P_SCORE_LAST4_WARD'] = m_data[p_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[g_score_last4_ward := 'G_SCORE_LAST4_WARD'] = m_data[g_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[p_score_all_ward := 'P_SCORE_ALL_WARD'] = m_data[p_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[g_score_all_ward := 'G_SCORE_ALL_WARD'] = m_data[g_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[p_score_last4_age_ward_precinct := 'P_SCORE_LAST4_AGE_WARD_PRECINCT'] = m_data[p_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[g_score_last4_age_ward_precinct := 'G_SCORE_LAST4_AGE_WARD_PRECINCT'] = m_data[g_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[p_score_all_age_ward_precinct := 'P_SCORE_ALL_AGE_WARD_PRECINCT'] = m_data[p_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[g_score_all_age_ward_precinct := 'G_SCORE_ALL_AGE_WARD_PRECINCT'] = m_data[g_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
ml_cat.interaction_features.extend([
    age_ward,
    age_precinct,
    age_party,
    p_score_last4_cat,
    g_score_last4_cat,
    p_score_all_cat,
    g_score_all_cat,
    p_score_last4_ward,
    g_score_last4_ward,
    p_score_all_ward,
    g_score_all_ward,
    p_score_last4_age_ward_precinct,
    g_score_last4_age_ward_precinct,
    p_score_all_age_ward_precinct,
    g_score_all_age_ward_precinct,
])

november_set = m_data[m_data['VOTED_NOV_LEVY'] == True]
y_pseudo = november_set['nov_for'] / november_set['total']

# Convert categorical columns to dummy variables
categorical_dummies = pd.get_dummies(november_set[ml_cat.category_features + ml_cat.interaction_features],
                                  drop_first=True, # Drop first category to avoid multicollinearity
                                  dtype=float)      # Convert to float for the model


preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False), ml_cat.category_features),
                ("high_card", TargetEncoder(), ml_cat.high_cardinality_features),
                ("num", StandardScaler(), ml_cat.numerical_features),
                ("interaction", TargetEncoder(), ml_cat.interaction_features),
            ]
        ) 

# Combine all features
X = preprocessor.fit_transform(november_set, november_set['nov_for_share'])
feature_names = preprocessor.get_feature_names_out()

y = y_pseudo  # Using the same target variable as before

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get predictions
y_pred_test = model.predict(X_test)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'against': np.abs(model.coef_[0]),
    'swing': np.abs(model.coef_[1]),
    'for': np.abs(model.coef_[2])
})
feature_importance['overall_importance'] = feature_importance[['against', 'swing', 'for']].mean(axis=1)
feature_importance = feature_importance.sort_values('overall_importance', ascending=False)

# Now plot the results
plot_regression_results(y_test, y_pred_test, 'Linear Regression')
plot_feature_importance(model, feature_names)

all_predictions = model.predict(X)

november_set['prediction_for_share'] = all_predictions
november_set['prediction_residual'] = y_pseudo - all_predictions

# Create KDE plot for prediction distributions
plt.figure(figsize=(12, 6))
sns.kdeplot(data=november_set['prediction_for_share'], label='Prediction Share', alpha=0.7)

actual_result = 0.4640  # November election result
std_dev = 0.07  # Standard deviation from November results
lean_margin = std_dev  # 1 standard deviation for lean (±7%)
strong_margin = std_dev * 2  # 2 standard deviations for strong (±14%)

# Add vertical line for actual result
plt.axvline(x=actual_result, color='red', linestyle='-', alpha=0.8, label='Actual November Result (46.40%)')

# Add threshold lines based on margins from actual result
plt.axvline(x=actual_result - strong_margin, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=actual_result - lean_margin, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=actual_result + lean_margin, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=actual_result + strong_margin, color='gray', linestyle='--', alpha=0.5)

# Add labels
plt.text(actual_result - strong_margin, plt.ylim()[1], f'\nStrongly Against\n(<{(actual_result-strong_margin)*100:.1f}%)', rotation=90, ha='right')
plt.text(actual_result - lean_margin, plt.ylim()[1], f'\nLean Against\n(<{(actual_result-lean_margin)*100:.1f}%)', rotation=90, ha='right')
plt.text(actual_result + lean_margin, plt.ylim()[1], f'\nLean For\n(>{(actual_result+lean_margin)*100:.1f}%)', rotation=90, ha='right')
plt.text(actual_result + strong_margin, plt.ylim()[1], f'\nStrongly For\n(>{(actual_result+strong_margin)*100:.1f}%)', rotation=90, ha='right')

plt.title('Distribution of Prediction Scores vs Actual November Result (±1σ and ±2σ)')
plt.xlabel('Prediction Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Recategorize predictions based on actual result
november_set['prediction_sentiment'] = pd.cut(
    november_set['prediction_for_share'],
    bins=[-np.inf, actual_result - strong_margin, actual_result - lean_margin, actual_result + lean_margin, actual_result + strong_margin, np.inf],
    labels=['strongly_against', 'lean_against', 'swing', 'lean_for', 'strongly_for']
)

# First, calculate the actual ward and precinct level results
ward_actual = november_set.groupby('WARD').agg({
    'nov_for_share': 'mean'
}).rename(columns={'nov_for_share': 'actual_ward_for_share'})

precinct_actual = november_set.groupby('PRECINCT_NAME').agg({
    'nov_for_share': 'mean'
}).rename(columns={'nov_for_share': 'actual_precinct_for_share'})

age_range_actual = november_set.groupby('AGE_RANGE').agg({
    'nov_for_share': 'mean'
}).rename(columns={'nov_for_share': 'actual_age_range_for_share'})

# Calculate predicted results at ward and precinct level
ward_predicted = november_set.groupby('WARD').agg({
    'prediction_for_share': 'mean'
}).rename(columns={'prediction_for_share': 'predicted_ward_for_share'})

precinct_predicted = november_set.groupby('PRECINCT_NAME').agg({
    'prediction_for_share': 'mean'
}).rename(columns={'prediction_for_share': 'predicted_precinct_for_share'})

age_range_predicted = november_set.groupby('AGE_RANGE').agg({
    'prediction_for_share': 'mean'
}).rename(columns={'prediction_for_share': 'predicted_age_range_for_share'})

# Combine actual and predicted results
ward_comparison = pd.merge(ward_actual, ward_predicted, left_index=True, right_index=True)
ward_comparison['difference'] = ward_comparison['predicted_ward_for_share'] - ward_comparison['actual_ward_for_share']

precinct_comparison = pd.merge(precinct_actual, precinct_predicted, left_index=True, right_index=True)
precinct_comparison['difference'] = precinct_comparison['predicted_precinct_for_share'] - precinct_comparison['actual_precinct_for_share']

age_range_comparison = pd.merge(age_range_actual, age_range_predicted, left_index=True, right_index=True)
age_range_comparison['difference'] = age_range_comparison['predicted_age_range_for_share'] - age_range_comparison['actual_age_range_for_share']

# Print comparisons
print("\nWard-Level Comparison:")
print("=" * 50)
print(ward_comparison.sort_values('difference', ascending=False).round(4))

print("\nPrecinct-Level Comparison:")
print("=" * 50)
print(precinct_comparison.sort_values('difference', ascending=False).round(4))

print("\nAge Range-Level Comparison:")
print("=" * 50)
print(age_range_comparison.sort_values('difference', ascending=False).round(4))

# Calculate summary statistics
print("\nSummary Statistics:")
print("=" * 50)
print("\nWard-Level Differences:")
print(f"Mean Absolute Error: {abs(ward_comparison['difference']).mean():.4f}")
print(f"Max Overprediction: {ward_comparison['difference'].max():.4f}")
print(f"Max Underprediction: {ward_comparison['difference'].min():.4f}")

print("\nPrecinct-Level Differences:")
print(f"Mean Absolute Error: {abs(precinct_comparison['difference']).mean():.4f}")
print(f"Max Overprediction: {precinct_comparison['difference'].max():.4f}")
print(f"Max Underprediction: {precinct_comparison['difference'].min():.4f}")

print("\nAge Range-Level Differences:")
print(f"Mean Absolute Error: {abs(age_range_comparison['difference']).mean():.4f}")
print(f"Max Overprediction: {age_range_comparison['difference'].max():.4f}")
print(f"Max Underprediction: {age_range_comparison['difference'].min():.4f}")

november_results = november_set[november_set['VOTED_NOV_LEVY'] == True].groupby('prediction_sentiment')['VOTED_NOV_LEVY'].sum().reset_index()
november_results['percent'] = (november_results['VOTED_NOV_LEVY'] / m_data['VOTED_NOV_LEVY'].sum()).round(4)
november_merge = november_set[['WARD', 'PRECINCT_NAME', 'AGE_RANGE','prediction_sentiment']].drop_duplicates()

november_quant = november_set['prediction_for_share'].quantile([0.15, 0.4, 0.6, 0.85])

may_voters = m_data[m_data['VOTED_MAY_LEVY'] == True]
may_merge = may_voters.merge(november_merge, on=['WARD', 'PRECINCT_NAME', 'AGE_RANGE'], how='inner', suffixes=('', '_november'))
may_results = may_merge.groupby('prediction_sentiment')['VOTED_MAY_LEVY'].sum().reset_index()
may_results['percent'] = (may_results['VOTED_MAY_LEVY'] / m_data['VOTED_MAY_LEVY'].sum()).round(4)
#
# may_voters = m_data[m_data['VOTED_MAY_LEVY'] == True]
# may_results['percent'] = (may_results['VOTED_MAY_LEVY'] / m_data['VOTED_MAY_LEVY'].sum()).round(4)





# Now you can merge this with your main dataset
# m_data = m_data.merge(
#     primary_ward_turnout_table,
#     on=['AGE_RANGE', 'WARD'],
#     how='left',
#     suffixes=('_primary', '_primary_ward')
# )
# m_data = m_data.merge(
#     general_ward_turnout_table,
#     on=['AGE_RANGE', 'WARD'],
#     how='left'
# )
# m_data = m_data.merge(
#     primary_precinct_turnout_table,
#     on=['AGE_RANGE', 'PRECINCT_NAME'],
#     how='left'
# )
# m_data = m_data.merge(
#     general_precinct_turnout_table,
#     on=['AGE_RANGE', 'PRECINCT_NAME'],
#     how='left'
# )
# m_data = m_data.merge(
#     primary_ward_party_turnout_table,
#     on=['PARTY_AFFILIATION', 'WARD'],
#     how='left'
# )
# m_data = m_data.merge(
#     general_ward_party_turnout_table,
#     on=['PARTY_AFFILIATION', 'WARD'],
#     how='left'
# )
# m_data = m_data.merge(
#     primary_precinct_party_turnout_table,
#     on=['PARTY_AFFILIATION', 'PRECINCT_NAME'],
#     how='left'
# )
# m_data = m_data.merge(
#     general_precinct_party_turnout_table,
#     on=['PARTY_AFFILIATION', 'PRECINCT_NAME'],
#     how='left'
# )




# november_voters = m_data[m_data['VOTED_NOV_LEVY'] == True]

# # Turnout by ward and age
# turnout_ward = pd.crosstab(
#     [m_data['AGE_RANGE'], m_data['WARD']], 
#     m_data['VOTED_NOV_LEVY'],
#     normalize='index'
# )[True].unstack()

# # Turnout by precinct and age
# turnout_precinct = pd.crosstab(
#     [m_data['AGE_RANGE'], m_data['PRECINCT_NAME']], 
#     m_data['VOTED_NOV_LEVY'],
#     normalize='index'
# )[True].unstack()

# for voter_idx, voter in m_data.iterrows():
#     ward = voter['WARD']
#     age_range = voter['AGE_RANGE']
#     precinct = voter['PRECINCT_NAME']
    
#     # Add city-wide context
#     m_data.loc[voter_idx, 'age_ward_city_share'] = age_ward_city.loc[age_range, ward]
#     m_data.loc[voter_idx, 'age_precinct_city_share'] = age_precinct_city.loc[age_range, precinct]
    
#     # Add local demographic context
#     m_data.loc[voter_idx, 'age_share_in_ward'] = age_ward_within.loc[age_range, ward]
#     m_data.loc[voter_idx, 'age_share_in_precinct'] = age_precinct_within.loc[age_range, precinct]
    
#     # Add historical turnout rates
#     m_data.loc[voter_idx, 'age_ward_turnout_rate'] = turnout_ward.loc[age_range, ward]



# m_data = pd.concat([m_data, pd.get_dummies(m_data['PARTY_AFFILIATION'], prefix='PARTY')], axis=1)
#
# m_data['AGE_RANGE_CAT'] = pd.Categorical(m_data['AGE_RANGE'], categories=vf_config.AGE_RANGE_SORTED, ordered=True)
# m_data['AGE_WARD'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['WARD'].astype(str)
# m_data['AGE_PRECINCT'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['PRECINCT_NAME'].astype(str)




# linear_model = FindlayPredictionModel(voterfile)
# decision_tree = FindlayDecisionTree(voterfile).run()
# decision_tree.run_sentiment_analysis()

# print("Running Monte Carlo simulation...")
# # Run Monte Carlo simulation with fewer iterations for testing
# mc_config = MonteCarloConfig(
#     n_simulations=100,  # Reduced number of simulations for testing
#     turnout_std=0.15,    # Adjust turnout variation
#     sentiment_std=0.2    # Adjust sentiment variation
# )
# mc_simulation = MonteCarloVoterSimulation(linear_model.data.model_data, mc_config)
# mc_simulation.run_simulation()
# mc_simulation.plot_distributions()
# mc_simulation.print_summary_statistics()
# mc_simulation_results = mc_simulation.get_voter_predictions()

# print("Generating prediction plots...")

# Prediction Plots

# all_voters_by_age = vote_prediction_by_age(linear_model.data.model_data, linear_model.data.config)
# november_voters = voterfile.model_data[voterfile.model_data[vf_cols.VOTED_NOV_LEVY] == 1]
# november_voters_by_age = vote_prediction_by_age(november_voters, linear_model.data.config)
# november_voters_by_ward = vote_prediction_by_ward(november_voters, linear_model.data.config)
# november_voters_by_precinct = vote_prediction_by_precinct(november_voters, linear_model.data.config)
# november_voter_count_by_age = november_voters.groupby([vf_cols.AGE_RANGE])[vf_cols.VOTER_ID].count().reset_index().rename(columns={vf_cols.VOTER_ID: 'VOTER_COUNT'})
# november_voter_count_by_age['PERCENT'] = (november_voter_count_by_age['VOTER_COUNT'] / november_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

# may_voters = voterfile.model_data[voterfile.model_data[vf_cols.VOTED_MAY_LEVY] == 1]
# may_voters_by_age = vote_prediction_by_age(may_voters, linear_model.data.config)
# may_voters_by_ward = vote_prediction_by_ward(may_voters, linear_model.data.config)
# may_voters_by_precinct = vote_prediction_by_precinct(may_voters, linear_model.data.config)
# may_voter_count_by_age = may_voters.groupby([vf_cols.AGE_RANGE])[vf_cols.VOTER_ID].count().reset_index().rename(columns={vf_cols.VOTER_ID: 'VOTER_COUNT'})
# may_voter_count_by_age['PERCENT'] = (may_voter_count_by_age['VOTER_COUNT'] / may_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

# ic(november_voters_by_ward.sum())

# plot_vote_share(november_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_TOTAL_COLS, 'Turnout: November Voters by Age')
# plot_vote_share(november_voters_by_ward, 'WARD', linear_model.config.PREDICTION_TOTAL_COLS, 'Turnout: November Voters by Ward')
# plot_vote_share(may_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_TOTAL_COLS, 'Turnout: May Voters by Age')
# plot_vote_share(may_voters_by_ward, 'WARD', linear_model.config.PREDICTION_TOTAL_COLS, 'Turnout: May Voters by Ward')

# plot_vote_share(november_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_LEVEL_COLS, 'Linear Model: November Voters by Age')
# plot_vote_share(november_voters_by_ward, 'WARD', linear_model.config.PREDICTION_LEVEL_COLS, 'Linear Model: November Voters by Ward')
# plot_vote_share(may_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_LEVEL_COLS, 'Linear Model: May Voters by Age')
# plot_vote_share(may_voters_by_ward, 'WARD', linear_model.config.PREDICTION_LEVEL_COLS, 'Linear Model: May Voters by Ward')


# nov_by_level = november_voters.groupby('vote_prediction')[vf_cols.VOTER_ID].count()
# nov_by_generic = november_voters.groupby('generic_vote_prediction')[vf_cols.VOTER_ID].count()
# nov_by_semi_generic = november_voters.groupby('semi_generic_prediction')[vf_cols.VOTER_ID].count()
#
# may_by_level = may_voters.groupby('vote_prediction')[vf_cols.VOTER_ID].count()
# may_by_generic = may_voters.groupby('generic_vote_prediction')[vf_cols.VOTER_ID].count()
# may_by_semi_generic = may_voters.groupby('semi_generic_prediction')[vf_cols.VOTER_ID].count()
#
# plot_pie_chart(nov_by_level, 'Linear Model: November Voters by Vote Prediction')
# plot_pie_chart(nov_by_semi_generic, 'Linear Model: November Voters by Semi-Generic Vote Prediction')
# plot_pie_chart(nov_by_generic, 'Linear Model: November Voters by Generic Vote Prediction')
#
# plot_pie_chart(may_by_level, 'Linear Model: May Voters by Vote Prediction')
# plot_pie_chart(may_by_semi_generic, 'Linear Model: May Voters by Semi-Generic Vote Prediction')
# plot_pie_chart(may_by_generic, 'Linear Model: May Voters by Generic Vote Prediction')


# # Exports

# november_voters_by_ward.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_november_voters_by_ward.csv', index=False)
# may_voters_by_ward.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_may_voters_by_ward.csv', index=False)
# november_voters_by_age.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_november_voters_by_age.csv', index=False)
# may_voters_by_age.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_may_voters_by_age.csv', index=False)
# november_voters_by_precinct.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_november_voters_by_precinct.csv', index=False)
# may_voters_by_precinct.to_csv(FilePaths.PREDICTION_FOLDER / 'linear_model_may_voters_by_precinct.csv', index=False)
