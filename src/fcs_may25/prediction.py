from pathlib import Path
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from icecream import ic
import matplotlib.pyplot as plt
import warnings

from data_loader import FindlayVoterFile, PREDICTION_FOLDER, IMAGE_PATH

voterfile = FindlayVoterFile()
config = voterfile.config


NOVEMBER_ELECTION = list(config.GENERAL_COLUMNS)[-1]
NOVEMBER_RESULTS_COLS = config.NOVEMBER_RESULTS_COLS
PREDICTION_LEVEL_COLS = config.PREDICTION_LEVEL_COLS
PREDICTION_TOTAL_COLS = config.PREDICTION_TOTAL_COLS
AGE_RANGE_SORTED = config.AGE_RANGE_SORTED

records = voterfile.data
election_results = voterfile.election_results


new_data = records.merge(election_results, right_on="precinct", left_on="PRECINCT_NAME", how="left")


ward_mapping = new_data.groupby('WARD')['nov_for_share'].mean().to_dict()
precinct_mapping = new_data.groupby('PRECINCT_NAME')['nov_for_share'].mean().to_dict()

new_data['AGE_RANGE_CAT'] = pd.Categorical(new_data['AGE_RANGE'], categories=AGE_RANGE_SORTED, ordered=True)
new_data['PARTY_CAT'] = pd.Categorical(new_data['PARTY_AFFILIATION'], categories=['D', 'I', 'R'], ordered=True)
new_data['AGE_WARD'] = new_data['AGE_RANGE_CAT'].astype(str) + '_' + new_data['WARD'].astype(str)
new_data['AGE_PRECINCT'] = new_data['AGE_RANGE_CAT'].astype(str) + '_' + new_data['PRECINCT_NAME'].astype(str)
new_data['AGE_PARTY'] = new_data['AGE_RANGE_CAT'].astype(str) + '_' + new_data['PARTY_CAT'].astype(str)

interaction_features = ["AGE_WARD", "AGE_PRECINCT", "AGE_PARTY"]
category_features = ["PARTY_CAT", "AGE_RANGE_CAT"]
high_cardinality_features = ["PRECINCT_NAME", "WARD"]
numerical_features = ["P_SCORE", "G_SCORE", "AGE"]
all_features = category_features + high_cardinality_features + numerical_features + interaction_features
new_data[all_features] = new_data[all_features]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features),
        ("high_card", TargetEncoder(), high_cardinality_features),
        ("num", StandardScaler(), numerical_features),
        ("interaction", TargetEncoder(), interaction_features),
    ]
)

def test_max_iter_impact(X, y, max_iter_values=[100, 500, 1000, 2000]):
    """Test the impact of different max_iter values on model accuracy."""
    results = []
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for max_iter in max_iter_values:
        # Create and fit the model
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        
        # Catch convergence warnings
        with warnings.catch_warnings(record=True) as w:
            model.fit(X_train, y_train)
            has_warning = any(issubclass(warn.category, UserWarning) for warn in w)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results.append({
            'max_iter': max_iter,
            'accuracy': accuracy,
            'converged': not has_warning,
            'n_iter': model.n_iter_[0] if hasattr(model, 'n_iter_') else None
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    print("\nImpact of max_iter on Model Performance:")
    print(results_df.to_string(index=False))
    
    return results_df

X = preprocessor.fit_transform(new_data, new_data["nov_for_share"])

# More granular thresholds based on your data distribution
y_pseudo_cat = pd.cut(
    new_data["nov_for_share"],
    bins=[-np.inf, 0.4, 0.5, 0.6, np.inf],
    labels=[0, 1, 2, 3]  # strongly against, lean against, lean for, strongly for
).astype(int)

# Run the test
max_iter_results = test_max_iter_impact(X, y_pseudo_cat) 

model = LogisticRegression(penalty=None, random_state=42, max_iter=500)
model.fit(X, y_pseudo_cat)

# Get feature names from the fitted preprocessor
feature_names = preprocessor.get_feature_names_out()

# Create a DataFrame with feature names and coefficients for all classes
feature_importance_all = pd.DataFrame({
    'feature': feature_names,
    'against': np.abs(model.coef_[0]),
    'swing': np.abs(model.coef_[1]),
    'for': np.abs(model.coef_[2])
})

# Calculate overall importance as the mean across all classes
feature_importance_all['overall_importance'] = feature_importance_all[['against', 'swing', 'for']].mean(axis=1)

# Sort by overall importance
feature_importance_all = feature_importance_all.sort_values('overall_importance', ascending=False)

# Display the top 15 most important features
ic("Top 15 most important features (overall):")
ic(feature_importance_all[['feature', 'overall_importance']].head(15))

# Visualize the top 15 features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_all['feature'][:15], feature_importance_all['overall_importance'][:15])
plt.xlabel('Feature Importance (Absolute Coefficient Value)')
plt.ylabel('Feature')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

y_pred = model.predict_proba(X)[:, 1]

new_data['P_for'] = y_pred

actual_for_share = new_data['nov_for_share'].mean()
actual_against_share = 1 - actual_for_share

model_weight = 0.3
election_weight = 0.7

new_data['weighted_prediction'] = (
    model_weight * new_data['P_for'] + 
    election_weight * new_data.apply(
        lambda row: (ward_mapping.get(row['WARD'], 0.5) + precinct_mapping.get(row['PRECINCT_NAME'], 0.5)) / 2, 
        axis=1
    )
)


# Create a 7-level prediction system
new_data['vote_prediction'] = pd.Series('strongly_against', index=new_data.index)  # Default

# Define thresholds for the 7 categories
strongly_against_threshold = new_data['weighted_prediction'].quantile(actual_against_share * 0.4)  # 40% of against voters
swing_against_threshold = new_data['weighted_prediction'].quantile(actual_against_share * 0.7)     # 30% of against voters
lean_against_threshold = new_data['weighted_prediction'].quantile(actual_against_share)            # 30% of against voters
swing_threshold = new_data['weighted_prediction'].quantile(0.5)                                    # Middle point
lean_for_threshold = new_data['weighted_prediction'].quantile(1 - actual_for_share * 0.3)          # 30% of for voters
swing_for_threshold = new_data['weighted_prediction'].quantile(1 - actual_for_share * 0.6)         # 30% of for voters
# Top 15% will be strongly_for

# Assign categories based on thresholds
new_data.loc[(new_data["weighted_prediction"] >= strongly_against_threshold) & 
             (new_data["weighted_prediction"] < swing_against_threshold), 'vote_prediction'] = 'swing_against'
new_data.loc[(new_data["weighted_prediction"] >= swing_against_threshold) & 
             (new_data["weighted_prediction"] < lean_against_threshold), 'vote_prediction'] = 'lean_against'
new_data.loc[(new_data["weighted_prediction"] >= lean_against_threshold) & 
             (new_data["weighted_prediction"] < swing_threshold), 'vote_prediction'] = 'swing'
new_data.loc[(new_data["weighted_prediction"] >= swing_threshold) & 
             (new_data["weighted_prediction"] < lean_for_threshold), 'vote_prediction'] = 'lean_for'
new_data.loc[(new_data["weighted_prediction"] >= lean_for_threshold) & 
             (new_data["weighted_prediction"] < swing_for_threshold), 'vote_prediction'] = 'swing_for'
new_data.loc[new_data["weighted_prediction"] >= swing_for_threshold, 'vote_prediction'] = 'strongly_for'

# Verify the distribution
ic("\nDistribution of 7-level predictions:")
ic(new_data['vote_prediction'].value_counts(normalize=True))


# Create a pie chart of the overall vote prediction counts
vote_counts = new_data['vote_prediction'].value_counts()
plt.figure(figsize=(12, 12))
vote_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=True, figsize=(12, 12))
plt.title('Overall Vote Prediction Distribution')
plt.ylabel('')
plt.show()

# Print the counts
ic("\nVote prediction counts:")
ic(vote_counts)

new_data['semi_generic_prediction'] = new_data['vote_prediction'].map({
    'strongly_for': 'for',
    'lean_for': 'for',
    'swing_for': 'swing',
    'swing_against': 'swing',
    'lean_against': 'against',
    'strongly_against': 'against'
})
new_data['generic_vote_prediction'] = new_data['vote_prediction'].map({
    'strongly_for': 'for',
    'lean_for': 'for',
    'swing_for': 'for',
    'swing_against': 'against',
    'lean_against': 'against',
    'strongly_against': 'against'
})

def add_total_for_vote_prediction(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        results['total_for_share'] = ((results['strongly_for'] + results['lean_for']) / results['Total'].sum()).round(4)
        results['total_against_share'] = ((results['strongly_against'] + results['lean_against']) / results['Total'].sum()).round(4)
        results['total_swing_share'] = ((results['swing_against']) / results['Total'].sum()).round(4)
        results.loc['Total', PREDICTION_TOTAL_COLS ] = results.sum(numeric_only=True).round(4)
        return results
    return wrapper

@add_total_for_vote_prediction
def vote_prediction_by_age(df):
    return pd.crosstab(
        index=df['AGE_RANGE'],
        columns=df['vote_prediction'],
        margins=True,
        margins_name="Total",
    )

@add_total_for_vote_prediction
def vote_prediction_by_ward(df):
    return pd.crosstab(
        index=df['WARD'],
        columns=df['vote_prediction'],
        margins=True,
        margins_name="Total",
    )

@add_total_for_vote_prediction
def vote_prediction_by_precinct(df):
    return pd.crosstab(
        index=df['PRECINCT_NAME'],
        columns=df['vote_prediction'],
        margins=True,
        margins_name="Total",
    )
def groupby_ward_vote_share(df):
    return df.reset_index().groupby('WARD').agg({
        'total_for_count': 'sum',
        'total_against_count': 'sum',
        'total_swing_count': 'sum',
    }).reset_index()

def groupby_age_vote_share(df):
    return df.reset_index().groupby('AGE_RANGE').agg({
        'total_for_count': 'sum',
        'total_against_count': 'sum',
        'total_swing_count': 'sum',
    }).reset_index()

def plot_vote_share(df, by: str, cols: list[str],  title: str):
    plot_df = df.reset_index().drop(columns=['Total'])
    plot_df = plot_df.iloc[:-1]
    ic(plot_df.columns.to_list())
    plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
    plt.show()

# def plot_vote_share_by_level(df, by: str, cols: list[str], title: str):
#     plot_df = df.reset_index()
#     ic(plot_df.columns.to_list())
#     plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
#     plt.show()

def plot_pie_chart(df, title: str):
    df.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, figsize=(8, 8), ylabel='', title=title)
    plt.title(title)
    plt.show()


all_voters_by_age = vote_prediction_by_age(new_data)
november_voters = new_data[new_data[NOVEMBER_ELECTION] == 1]
november_voters_by_age = vote_prediction_by_age(november_voters)
november_voters_by_ward = vote_prediction_by_ward(november_voters)
november_voters_by_precinct = vote_prediction_by_precinct(november_voters)
november_voter_count_by_age = november_voters.groupby(['AGE_RANGE'])['SOS_VOTERID'].count().reset_index().rename(columns={'SOS_VOTERID': 'VOTER_COUNT'})
november_voter_count_by_age['PERCENT'] = (november_voter_count_by_age['VOTER_COUNT'] / november_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

may_voters = new_data[new_data['VOTED_MAY_LEVY'] == 1]
may_voters_by_age = vote_prediction_by_age(may_voters)
may_voters_by_ward = vote_prediction_by_ward(may_voters)
may_voters_by_precinct = vote_prediction_by_precinct(may_voters)
may_voter_count_by_age = may_voters.groupby(['AGE_RANGE'])['SOS_VOTERID'].count().reset_index().rename(columns={'SOS_VOTERID': 'VOTER_COUNT'})
may_voter_count_by_age['PERCENT'] = (may_voter_count_by_age['VOTER_COUNT'] / may_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

ic(november_voters_by_ward.sum())

plot_vote_share(november_voters_by_age, 'AGE_RANGE', cols=PREDICTION_TOTAL_COLS, title='November Voters by Age')
plot_vote_share(november_voters_by_ward, 'WARD', cols=PREDICTION_TOTAL_COLS, title='November Voters by Ward')
plot_vote_share(may_voters_by_age, 'AGE_RANGE', PREDICTION_TOTAL_COLS, 'May Voters by Age')
plot_vote_share(may_voters_by_ward, 'WARD', PREDICTION_TOTAL_COLS, 'May Voters by Ward')

plot_vote_share(november_voters_by_age, 'AGE_RANGE', PREDICTION_LEVEL_COLS, 'November Voters by Age')
plot_vote_share(november_voters_by_ward, 'WARD', PREDICTION_LEVEL_COLS, 'November Voters by Ward')
plot_vote_share(may_voters_by_age, 'AGE_RANGE', PREDICTION_LEVEL_COLS, 'May Voters by Age')
plot_vote_share(may_voters_by_ward, 'WARD', PREDICTION_LEVEL_COLS, 'May Voters by Ward')


nov_by_level = november_voters.groupby('vote_prediction')['SOS_VOTERID'].count()
nov_by_generic = november_voters.groupby('generic_vote_prediction')['SOS_VOTERID'].count()
nov_by_semi_generic = november_voters.groupby('semi_generic_prediction')['SOS_VOTERID'].count()

may_by_level = may_voters.groupby('vote_prediction')['SOS_VOTERID'].count()
may_by_generic = may_voters.groupby('generic_vote_prediction')['SOS_VOTERID'].count()
may_by_semi_generic = may_voters.groupby('semi_generic_prediction')['SOS_VOTERID'].count()

plot_pie_chart(nov_by_level, 'November Voters by Vote Prediction')
plot_pie_chart(nov_by_semi_generic, 'November Voters by Semi-Generic Vote Prediction')
plot_pie_chart(nov_by_generic, 'November Voters by Generic Vote Prediction')

plot_pie_chart(may_by_level, 'May Voters by Vote Prediction')
plot_pie_chart(may_by_semi_generic, 'May Voters by Semi-Generic Vote Prediction')
plot_pie_chart(may_by_generic, 'May Voters by Generic Vote Prediction')




""" DECISION TREE"""
new_data['vote_decision'] = (new_data['nov_for_share'] >= 0.5).astype(int)
decision_features = [
    'AGE_RANGE',
    'PARTY_CAT',
    'WARD',
    'PRECINCT_NAME',
    'AGE_WARD',
    'AGE_PRECINCT',
    'AGE_PARTY',
    'P_SCORE',
    'G_SCORE',
    'AGE'
]

# Prepare X and y
X_decision = new_data[decision_features].copy()
y_decision = new_data['vote_decision']

le = LabelEncoder()
X_decision['AGE_RANGE'] = le.fit_transform(X_decision['AGE_RANGE'])
X_decision['PARTY_CAT'] = le.fit_transform(X_decision['PARTY_CAT'])
X_decision['WARD'] = le.fit_transform(X_decision['WARD'])
X_decision['AGE_WARD'] = le.fit_transform(X_decision['AGE_WARD'])
X_decision['AGE_PRECINCT'] = le.fit_transform(X_decision['AGE_PRECINCT'])
X_decision['AGE_PARTY'] = le.fit_transform(X_decision['AGE_PARTY'])
X_decision['PRECINCT_NAME'] = le.fit_transform(X_decision['PRECINCT_NAME'])

tree = DecisionTreeClassifier(
    max_depth=5,  # Limit depth for interpretability
    min_samples_split=100,  # Minimum samples required to split
    min_samples_leaf=50,  # Minimum samples required at each leaf
    random_state=42
)

tree.fit(X_decision, y_decision)

plt.figure(figsize=(20,10))
plot_tree(
    tree,
    feature_names=decision_features,
    class_names=['Against', 'For'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

feature_importance = pd.DataFrame({
    'feature': decision_features,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

ic("\nFeature Importances:")
ic(feature_importance)

# Function to analyze decision paths
def analyze_decision_paths(tree, X, feature_names):
    # Get the decision paths
    paths = tree.decision_path(X)
    
    # Get leaf node assignments
    leaves = tree.apply(X)
    
    # Create a DataFrame to store path information
    path_info = []
    
    for i in range(len(X)):
        # Get the path for this sample
        path = paths.getrow(i)
        path_nodes = path.indices
        
        # Get the leaf node
        leaf = leaves[i]
        
        # Get the prediction
        pred = tree.predict([X.iloc[i]])[0]
        
        # Store the information
        path_info.append({
            'sample_index': i,
            'path_nodes': path_nodes,
            'leaf_node': leaf,
            'prediction': pred
        })
    
    return path_info

path_info = analyze_decision_paths(tree, X_decision, decision_features)
# Print some example paths
ic("\nExample Decision Paths:")
for i in range(min(5, len(path_info))):
    ic(f"\nSample {i}:")
    ic(f"Prediction: {'For' if path_info[i]['prediction'] == 1 else 'Against'}")
    ic("Decision path:")
    for node in path_info[i]['path_nodes']:
        if node != path_info[i]['leaf_node']:  # Skip leaf nodes
            feature = decision_features[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            value = X_decision.iloc[i][feature]
            direction = ">=" if value >= threshold else "<"
            ic(f"  {feature} {direction} {threshold:.2f}")

november_voters_by_ward.to_csv(PREDICTION_FOLDER / 'november_voters_by_ward.csv', index=False)
may_voters_by_ward.to_csv(PREDICTION_FOLDER / 'may_voters_by_ward.csv', index=False)
november_voters_by_age.to_csv(PREDICTION_FOLDER / 'november_voters_by_age.csv', index=False)
may_voters_by_age.to_csv(PREDICTION_FOLDER / 'may_voters_by_age.csv', index=False)
november_voters_by_precinct.to_csv(PREDICTION_FOLDER / 'november_voters_by_precinct.csv', index=False)
may_voters_by_precinct.to_csv(PREDICTION_FOLDER / 'may_voters_by_precinct.csv', index=False)

def create_sentiment_categories(prediction_probs, confidence_thresholds=None):
    """
    Create nuanced sentiment categories based on prediction probabilities.
    
    Args:
        prediction_probs: Array of prediction probabilities
        confidence_thresholds: Dictionary of thresholds for different sentiment levels
                             Default: {
                                 'strongly_against': 0.2,
                                 'moderately_against': 0.35,
                                 'slightly_against': 0.45,
                                 'neutral': 0.55,
                                 'slightly_for': 0.65,
                                 'moderately_for': 0.8,
                                 'strongly_for': 1.0
                             }
    """
    if confidence_thresholds is None:
        confidence_thresholds = {
            'strongly_against': 0.2,
            'moderately_against': 0.35,
            'slightly_against': 0.45,
            'neutral': 0.55,
            'slightly_for': 0.65,
            'moderately_for': 0.8,
            'strongly_for': 1.0
        }
    
    categories = []
    for prob in prediction_probs:
        if prob <= confidence_thresholds['strongly_against']:
            categories.append('strongly_against')
        elif prob <= confidence_thresholds['moderately_against']:
            categories.append('moderately_against')
        elif prob <= confidence_thresholds['slightly_against']:
            categories.append('slightly_against')
        elif prob <= confidence_thresholds['neutral']:
            categories.append('neutral')
        elif prob <= confidence_thresholds['slightly_for']:
            categories.append('slightly_for')
        elif prob <= confidence_thresholds['moderately_for']:
            categories.append('moderately_for')
        else:
            categories.append('strongly_for')
    
    return categories

# In your model prediction section:
prediction_probs = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class
sentiment_categories = create_sentiment_categories(prediction_probs)

# Create dummy variables for the new categories
sentiment_dummies = pd.get_dummies(sentiment_categories, prefix='sentiment')

# Add the sentiment categories to your DataFrame
new_data = pd.concat([new_data, sentiment_dummies], axis=1)

# Print distribution of sentiments
print("\nDistribution of Voter Sentiments:")
print(new_data['sentiment'].value_counts(normalize=True).round(3))

# Visualize the sentiment distribution
plt.figure(figsize=(10, 6))
new_data['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution of Voter Sentiments')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.close()