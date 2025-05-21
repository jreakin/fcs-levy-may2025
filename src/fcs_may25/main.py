# import marimo as mo
from category_encoders import TargetEncoder
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.compose import ColumnTransformer
import pandas as pd
import november_model as pe

vf = pe.voterfile
print(list(pe.m_data.columns))
X = pe.m_data[list(set(pe.ml_cat.category_features + pe.ml_cat.interaction_features + pe.ml_cat.numerical_features))]
y_turnout = pe.m_data["VOTED_NOV_LEVY"]
y_vote = pe.m_data["nov_for_share"]
y_pseudo = pe.m_data['nov_for'] / pe.m_data['total']
preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False), pe.ml_cat.category_features),
                ("high_card", TargetEncoder(), pe.ml_cat.high_cardinality_features),
                ("num", StandardScaler(), pe.ml_cat.numerical_features),
                ("interaction", TargetEncoder(), pe.ml_cat.interaction_features),
            ]
        )

X_encoded = preprocessor.fit_transform(X, y_turnout)
feature_names = preprocessor.get_feature_names_out()
feature_df = pd.DataFrame(X_encoded, columns=feature_names, index=pe.m_data.index)

# Split data for turnout model (all individuals)
X_train_turnout, X_test_turnout, y_turnout_train, y_turnout_test = train_test_split(
    X_encoded, y_turnout, test_size=0.2, random_state=42
)

# Filter voters for vote preference model
voters = pe.m_data[pe.m_data["VOTED_NOV_LEVY"] == 1].index
X_voters = X_encoded[voters]
y_vote_voters = y_vote[voters]

# Split voter data for vote preference
X_train_vote, X_test_vote, y_vote_train, y_vote_test = train_test_split(
    X_voters, y_vote_voters, test_size=0.2, random_state=42
)

# Train turnout model (LogisticRegression) on all individuals
turnout_model = LogisticRegression(random_state=42, max_iter=1000)
turnout_model.fit(X_train_turnout, y_turnout_train)

# Train vote preference model (LinearRegression) on voters
vote_model = LinearRegression(fit_intercept=True)
vote_model.fit(X_train_vote, y_vote_train)

# Predict for all individuals
turnout_prob = turnout_model.predict_proba(X_encoded)[:, 1]
P_for = vote_model.predict(X_encoded)
pe.m_data["turnout_prob"] = turnout_prob
pe.m_data["P_for"] = np.clip(P_for, 0, 1)

# Filter for November voters
november_set = pe.m_data[pe.m_data['VOTED_NOV_LEVY'] == 1]

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
    'P_for': 'mean'
}).rename(columns={'P_for': 'predicted_ward_for_share'})

precinct_predicted = november_set.groupby('PRECINCT_NAME').agg({
    'P_for': 'mean'
}).rename(columns={'P_for': 'predicted_precinct_for_share'})

age_range_predicted = november_set.groupby('AGE_RANGE').agg({
    'P_for': 'mean'
}).rename(columns={'P_for': 'predicted_age_range_for_share'})

# Recategorize predictions with tighter margins
actual_result = 0.4640  # November election result
std_dev = 0.07  # Standard deviation from November results
lean_margin = std_dev * 0.75  # Reduced margin to better match actual distribution
strong_margin = std_dev * 1.5  # Reduced margin to better match actual distribution


# Modify the adjust_predictions function
def adjust_predictions(group):
    if len(group) == 0 or group["P_for"].isna().all():
        return group
    target_for_share = group["nov_for_share"].iloc[0]
    mean_p_for = group["P_for"].mean()
    if mean_p_for != 0:
        # Add a scaling factor to better match November results
        scaling_factor = 0.46 / mean_p_for  # Using the actual November result
        group["P_for"] = np.clip(group["P_for"] * scaling_factor, 0, 1)
    group["P_against"] = 1 - group["P_for"]
    group["likely_vote"] = np.where(group["P_for"] > actual_result, "for", "against")
    return group

# Apply the adjusted predictions
pe.m_data.loc[voters] = pe.m_data.loc[voters].groupby(["ward", "precinct"]).apply(adjust_predictions).reset_index(drop=True)


# Update the prediction sentiment categorization
november_set['prediction_sentiment'] = pd.cut(
    november_set['P_for'], 
    bins=[-np.inf, pe.thresholds[0.2], pe.thresholds[0.40], pe.thresholds[0.60], pe.thresholds[0.8], np.inf],
    labels=['strongly_against', 'lean_against', 'swing', 'lean_for', 'strongly_for']
)

# Create a mapping dictionary
sentiment_map = {
    'strongly_for': 'for',
    'lean_for': 'for',
    'swing': 'swing',
    'lean_against': 'against',
    'strongly_against': 'against'
}

# Apply the mapping to create a new column
november_set['prediction_sentiment_broad'] = november_set['prediction_sentiment'].map(sentiment_map)


# Identify non-voters who should've voted (turnout_prob > 0.5)
non_voters = pe.m_data[pe.m_data["VOTED_NOV_LEVY"] == 0].index
pe.m_data["should_have_voted"] = (pe.m_data["turnout_prob"] > 0.5) & (pe.m_data["VOTED_NOV_LEVY"] == 0)

# Aggregate turnout and hypothetical voting
precinct_turnout = pe.m_data.groupby(["ward", "precinct"]).agg(
    actual_voters=("VOTED_NOV_LEVY", "sum"),
    estimated_voters=("turnout_prob", "sum"),
    should_have_voted_count=("should_have_voted", "sum"),
    hypothetical_for_votes=("P_for", lambda x: (x[pe.m_data["should_have_voted"]]).sum())
).reset_index()
precinct_turnout[precinct_turnout.columns[2:]] = precinct_turnout[precinct_turnout.columns[2:]].round().astype(int)
precinct_turnout['diff'] = precinct_turnout['actual_voters'] - precinct_turnout['estimated_voters']
# precinct_turnout = precinct_turnout[['ward', 'precinct', 'actual_voters', 'estimated_voters', 'diff', 'should_have_voted_count', 'hypothetical_for_votes']]
# Compute losses
train_pred_turnout = turnout_model.predict_proba(X_train_turnout)[:, 1]
test_pred_turnout = turnout_model.predict_proba(X_test_turnout)[:, 1]
train_pred_vote = vote_model.predict(X_train_vote)
test_pred_vote = vote_model.predict(X_test_vote)
train_loss_turnout = log_loss(y_turnout_train, train_pred_turnout)
test_loss_turnout = log_loss(y_turnout_test, test_pred_turnout)
train_mse_vote = mean_squared_error(y_vote_train, train_pred_vote)
test_mse_vote = mean_squared_error(y_vote_test, test_pred_vote)

# 1. First, let's look at feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(vote_model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
# plt.figure(figsize=(12, 6))
# sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
# plt.title('Top 20 Most Important Features')
# plt.tight_layout()
# plt.show()


# 4. Let's analyze the relationship between age and other important features
# First, get the top 5 most important features
top_features = feature_importance.head(5)['feature'].tolist()

# # Create a pairplot for these features
# plt.figure(figsize=(15, 10))
# sns.pairplot(
#     november_set[['AGE_RANGE', 'P_for']],
#     hue='AGE_RANGE',
#     diag_kind='kde'
# )
# plt.suptitle('Relationship Between Age and Predictions', y=1.02)
# plt.show()

# # 5. Let's look at the actual coefficients for age-related features
# age_features = feature_importance[feature_importance['feature'].str.contains('AGE')]
# print("\nAge-related feature coefficients:")
# print(age_features)

# 6. Create a heatmap of predictions by age and ward
age_ward_predictions = pd.pivot_table(
    november_set,
    values='P_for',
    index='AGE_RANGE',
    columns='WARD',
    aggfunc='mean'
)

age_precinct_predictions = pd.pivot_table(
    november_set,
    values='P_for',
    index='AGE_RANGE',
    columns='PRECINCT_NAME',
    aggfunc='mean'
)
# plt.figure(figsize=(12, 8))
# sns.heatmap(age_ward_predictions, annot=True, cmap='RdYlBu', center=0.5)
# plt.title('Average Predictions by Age Range and Ward')
# plt.tight_layout()
# plt.show()

# Additional Analysis
def create_crosstab(index_cols: list[str] | str, columns: list[str] | str, df: pd.DataFrame):
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    else:
        index_cols = [df[x] for x in index_cols]
    if isinstance(columns, str):
        _columns = [columns]
    else:
        _columns = [df[x] for x in columns]
    return pd.crosstab(
        index=index_cols,
        columns=_columns,
        margins=True,
        normalize='index'
    ).round(4)

city_age_range_crostab = create_crosstab(
    index_cols=['WARD'],
    columns=['AGE_RANGE'],
    df=pe.m_data
)

may_age_range_crostab = create_crosstab(
    index_cols=['WARD'],
    columns=['AGE_RANGE'],
    df=pe.may
)

may_sentiment = november_set[november_set['VOTED_MAY_LEVY'] == 1]
may_prediction_setiment = pd.crosstab(
    index=may_sentiment['AGE_RANGE'],
    columns=may_sentiment['prediction_sentiment_broad'],
    margins=True,
    normalize='index'
).round(4)

# pe.m_data[pe.m_data['VOTED_NOV_LEVY'] == True]['AGE_RANGE'].value_counts(normalize=True).sort_index()
# pe.m_data[pe.m_data['VOTED_MAY_LEVY'] == True]['AGE_RANGE'].value_counts(normalize=True).sort_index().plot(kind='pie', autopct='%1.1f%%')
#
# # Analyze 65+ voter patterns
# senior_voters = november_set[november_set['AGE_RANGE'].isin(['65-74', '75+'])]
#
# # Calculate November voting patterns for seniors
# senior_vote_analysis = senior_voters.groupby('AGE_RANGE').agg({
#     'nov_for_share': ['mean', 'count'],
#     'P_for': ['mean', 'count']
# }).round(4)
#
# print("\nNovember Voting Patterns for 65+ Voters:")
# print("=" * 50)
# print(senior_vote_analysis)
#
# # Calculate ward-level patterns for seniors
# senior_ward_analysis = senior_voters.groupby(['WARD', 'AGE_RANGE']).agg({
#     'nov_for_share': ['mean', 'count'],
#     'P_for': ['mean', 'count']
# }).round(4)
#
# print("\nWard-Level Analysis for 65+ Voters:")
# print("=" * 50)
# print(senior_ward_analysis)
#
# # Calculate the overall impact of senior voters
# total_nov_votes = november_set['VOTED_NOV_LEVY'].sum()
# senior_nov_votes = senior_voters['VOTED_NOV_LEVY'].sum()
# senior_for_votes = (senior_voters['nov_for_share'] * senior_voters['VOTED_NOV_LEVY']).sum()
#
# print("\nSenior Voter Impact Analysis:")
# print("=" * 50)
# print(f"Total November Votes: {total_nov_votes}")
# print(f"Senior November Votes: {senior_nov_votes}")
# print(f"Senior Vote Share: {(senior_nov_votes/total_nov_votes*100):.1f}%")
# print(f"Senior For Votes: {senior_for_votes:.0f}")
# print(f"Senior For Share: {(senior_for_votes/senior_nov_votes*100):.1f}%")
#
# # Visualize senior voting patterns
# plt.figure(figsize=(12, 6))
# senior_voters.groupby('AGE_RANGE')['nov_for_share'].mean().plot(kind='bar')
# plt.axhline(y=0.4640, color='red', linestyle='--', label='Overall November Result (46.40%)')
# plt.title('November For Share by Senior Age Group')
# plt.ylabel('For Share')
# plt.xlabel('Age Range')
# plt.legend()
# plt.show()
#
# # Compare senior voting patterns with May early voting
# may_seniors = pe.m_data[pe.m_data['VOTED_MAY_LEVY'] == 1]
# may_senior_share = may_seniors[may_seniors['AGE_RANGE'].isin(['65-74', '75+'])].shape[0] / may_seniors.shape[0]
#
# print("\nMay vs November Senior Voter Comparison:")
# print("=" * 50)
# print(f"May Early Vote Senior Share: {may_senior_share*100:.1f}%")
# print(f"November Senior Share: {(senior_nov_votes/total_nov_votes*100):.1f}%")
#
# # Senior Voter Analysis
# print("\nSenior Voter Analysis (65+):")
# print("=" * 50)
#
# # Filter for senior voters
# senior_voters = november_set[november_set['AGE_RANGE'].isin(['65-74', '75+'])]
#
# # Calculate November voting patterns
# total_nov_votes = november_set['VOTED_NOV_LEVY'].sum()
# senior_nov_votes = senior_voters['VOTED_NOV_LEVY'].sum()
# senior_for_votes = (senior_voters['nov_for_share'] * senior_voters['VOTED_NOV_LEVY']).sum()
#
# print(f"Total November Votes: {total_nov_votes}")
# print(f"Senior November Votes: {senior_nov_votes}")
# print(f"Senior Vote Share: {(senior_nov_votes/total_nov_votes*100):.1f}%")
# print(f"Senior For Votes: {senior_for_votes:.0f}")
# print(f"Senior For Share: {(senior_for_votes/senior_nov_votes*100):.1f}%")
#
# # Compare with May early voting
# may_seniors = pe.m_data[pe.m_data['VOTED_MAY_LEVY'] == 1]
# may_senior_share = may_seniors[may_seniors['AGE_RANGE'].isin(['65-74', '75+'])].shape[0] / may_seniors.shape[0]
#
# print("\nMay vs November Senior Voter Comparison:")
# print(f"May Early Vote Senior Share: {may_senior_share*100:.1f}%")
# print(f"November Senior Share: {(senior_nov_votes/total_nov_votes*100):.1f}%")
#
# # Ward-level analysis for seniors
# senior_ward_analysis = senior_voters.groupby(['WARD', 'AGE_RANGE']).agg({
#     'nov_for_share': ['mean', 'count'],
#     'P_for': ['mean', 'count']
# }).round(4)
#
# print("\nWard-Level Analysis for 65+ Voters:")
# print(senior_ward_analysis)
