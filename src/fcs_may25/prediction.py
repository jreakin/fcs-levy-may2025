import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from icecream import ic
import matplotlib.pyplot as plt
import warnings

import protocols as p_

from data_loader import (
    FindlayVoterFile,
    FilePaths, 
    FindlayVoterFileColumns as vf_cols, 
    FindlayMLModelCategories as ml_cat,
    NovemberResultsColumns as nov_results
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
    print("=" * 50)
    print(results_df.to_string(index=False))
    print("-" * 50)
    
    return results_df


def add_total_for_vote_prediction(func):
    def wrapper(df, config, *args, **kwargs):
        results = func(df, config, *args, **kwargs)
        results['total_for_share'] = ((results['strongly_for'] + results['lean_for']) / results['Total'].sum()).round(4)
        results['total_against_share'] = ((results['strongly_against'] + results['lean_against']) / results['Total'].sum()).round(4)
        results['total_swing_share'] = ((results['swing_against']) / results['Total'].sum()).round(4)
        results.loc['Total', config.PREDICTION_TOTAL_COLS] = results.sum(numeric_only=True).round(4)
        return results
    return wrapper

@add_total_for_vote_prediction
def vote_prediction_by_age(df, config):
    return pd.crosstab(
        index=df['AGE_RANGE'],
        columns=df['vote_prediction'],
        margins=True,
        margins_name="Total",
    )

@add_total_for_vote_prediction
def vote_prediction_by_ward(df, config):
    return pd.crosstab(
        index=df['WARD'],
        columns=df['vote_prediction'],
        margins=True,
        margins_name="Total",
    )

@add_total_for_vote_prediction
def vote_prediction_by_precinct(df, config):
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

def groupby_age_vote_share(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index().groupby('AGE_RANGE').agg({
        'total_for_count': 'sum',
        'total_against_count': 'sum',
        'total_swing_count': 'sum',
    }).reset_index()

def plot_vote_share(df: pd.DataFrame, by: str, cols: list[str],  title: str):
    plot_df = df.reset_index().drop(columns=['Total'])
    plot_df = plot_df.iloc[:-1]
    ic(plot_df.columns.to_list())
    plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
    plt.savefig(FilePaths.IMAGE_PATH / f'{title}.png')
    plt.show()

# def plot_vote_share_by_level(df, by: str, cols: list[str], title: str):
#     plot_df = df.reset_index()
#     ic(plot_df.columns.to_list())
#     plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
#     plt.show()
def plot_pie_chart(df: pd.DataFrame, title: str):
    df.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, figsize=(8, 8), ylabel='', title=title)
    plt.title(title)
    plt.savefig(FilePaths.IMAGE_PATH / f'{title}.png')
    plt.show()
    

class FindlayPredictionModel:
    data: p_.ModelDataStartingPoint
    ward_mapping: dict[str, float]
    precinct_mapping: dict[str, float]
    preprocessor: ColumnTransformer
    X_pseudo: pd.DataFrame
    y_pseudo: pd.Series
    model: LogisticRegression
    feature_names: list[str]
    feature_importance: pd.DataFrame

    def __init__(self, data: p_.ModelDataStartingPoint):
        self.data = data
        self.config = data.config
        self.ward_mapping = data.model_data.groupby(vf_cols.WARD)[nov_results.FOR_SHARE].mean().to_dict()
        self.precinct_mapping = data.model_data.groupby(vf_cols.PRECINCT_NAME)[nov_results.FOR_SHARE].mean().to_dict()

    
    def add_feature_columns(self, category_cols: p_.LinearModelFeatureColumns = ml_cat, voterfile_cols: p_.VoterFileColumns = vf_cols):
        _model_data = self.data.model_data
        _model_data[category_cols.AGE_RANGE_CAT] = pd.Categorical(_model_data[vf_cols.AGE_RANGE], categories=self.config.AGE_RANGE_SORTED, ordered=True)
        _model_data[category_cols.PARTY_CAT] = pd.Categorical(_model_data[vf_cols.PARTY_AFFILIATION], categories=['D', 'I', 'R'], ordered=True)
        _model_data[category_cols.AGE_WARD] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.WARD].astype(str)
        _model_data[category_cols.AGE_PRECINCT] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.PRECINCT_NAME].astype(str)
        _model_data[category_cols.AGE_PARTY] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.PARTY_AFFILIATION].astype(str)
        self.data.model_data = _model_data
        return self
    
    def create_preprocessor(self, feature_lists: p_.LinearModelFeatureLists = ml_cat):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False), feature_lists.category_features),
                ("high_card", TargetEncoder(), feature_lists.high_cardinality_features),
                ("num", StandardScaler(), feature_lists.numerical_features),
                ("interaction", TargetEncoder(), feature_lists.interaction_features),
            ]
        )
        return self

    def fit_pseudo_preprocessor(self) -> tuple[pd.DataFrame, pd.Series]:
        self.X_pseudo= self.preprocessor.fit_transform(self.data.model_data, self.data.model_data[nov_results.FOR_SHARE])
        # More granular thresholds based on your data distribution
        self.y_pseudo = pd.cut(
            self.data.model_data[nov_results.FOR_SHARE],
            bins=[-np.inf, 0.4, 0.5, 0.6, np.inf],
            labels=[0, 1, 2, 3]  # strongly against, lean against, lean for, strongly for
        ).astype(int)
        return self.X_pseudo, self.y_pseudo
    
    def run(self):
        self.add_feature_columns()
        self.create_preprocessor()
        self.fit_pseudo_preprocessor()
        self.run_max_iter_test()
        self.fit_model()
        self.get_feature_names()
        self.predict_probability()
        self.create_weighted_prediction()
        return self
    
    def run_max_iter_test(self):
        X, y = self.fit_pseudo_preprocessor()
        return test_max_iter_impact(X, y)
    
    def fit_model(self):
        self.model = LogisticRegression(penalty=None, random_state=42, max_iter=500)
        self.model.fit(self.X_pseudo, self.y_pseudo)
        return self.model
    
    def get_feature_names(self):
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'against': np.abs(self.model.coef_[0]),
            'swing': np.abs(self.model.coef_[1]),
            'for': np.abs(self.model.coef_[2])
        })
        # Calculate overall importance as the mean across all classes
        self.feature_importance['overall_importance'] = self.feature_importance[['against', 'swing', 'for']].mean(axis=1)

        # Sort by overall importance
        self.feature_importance = self.feature_importance.sort_values('overall_importance', ascending=False)

        # Display the top 15 most important features
        print("Linear Model: Top 15 most important features (overall):")
        print("=" * 50)
        print(self.feature_importance[['feature', 'overall_importance']].head(15).to_markdown(index=False))
        print("-" * 50)

        # Visualize the top 15 features
        plt.figure(figsize=(12, 8))
        plt.barh(self.feature_importance['feature'][:15], self.feature_importance['overall_importance'][:15])
        plt.xlabel('Feature Importance (Absolute Coefficient Value)')
        plt.ylabel('Feature')
        plt.title('Linear Model: Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig(FilePaths.IMAGE_PATH / 'feature_importance.png')
        plt.close()

        return self.feature_names
    
    def predict_probability(self):
        y_pred = self.model.predict_proba(self.X_pseudo)[:, 1]
        self.data.model_data['P_for'] = y_pred
        return self.data.model_data
    

    def create_weighted_prediction(self):
        actual_for_share = self.data.model_data[nov_results.FOR_SHARE].mean()
        actual_against_share = 1 - actual_for_share

        model_weight = 0.3
        election_weight = 0.7

        self.data.model_data['weighted_prediction'] = (
            model_weight * self.data.model_data['P_for'] + 
            election_weight * self.data.model_data.apply(
                lambda row: (self.ward_mapping.get(row[vf_cols.WARD], 0.5) + self.precinct_mapping.get(row[vf_cols.PRECINCT_NAME], 0.5)) / 2, 
                axis=1
            )
        )   

        # Create a 7-level prediction system
        self.data.model_data['vote_prediction'] = pd.Series('strongly_against', index=self.data.model_data.index)  # Default

        # Define thresholds for the 7 categories
        strongly_against_threshold = self.data.model_data['weighted_prediction'].quantile(actual_against_share * 0.4)  # 40% of against voters
        swing_against_threshold = self.data.model_data['weighted_prediction'].quantile(actual_against_share * 0.7)     # 30% of against voters
        lean_against_threshold = self.data.model_data['weighted_prediction'].quantile(actual_against_share)            # 30% of against voters
        swing_threshold = self.data.model_data['weighted_prediction'].quantile(0.5)                                    # Middle point
        lean_for_threshold = self.data.model_data['weighted_prediction'].quantile(1 - actual_for_share * 0.3)          # 30% of for voters
        swing_for_threshold = self.data.model_data['weighted_prediction'].quantile(1 - actual_for_share * 0.6)         # 30% of for voters
        # Top 15% will be strongly_for

        # Assign categories based on thresholds
        self.data.model_data.loc[(self.data.model_data["weighted_prediction"] >= strongly_against_threshold) & 
             (self.data.model_data["weighted_prediction"] < swing_against_threshold), 'vote_prediction'] = 'swing_against'
        self.data.model_data.loc[(self.data.model_data["weighted_prediction"] >= swing_against_threshold) & 
             (self.data.model_data["weighted_prediction"] < lean_against_threshold), 'vote_prediction'] = 'lean_against'
        self.data.model_data.loc[(self.data.model_data["weighted_prediction"] >= lean_against_threshold) & 
             (self.data.model_data["weighted_prediction"] < swing_threshold), 'vote_prediction'] = 'swing'
        self.data.model_data.loc[(self.data.model_data["weighted_prediction"] >= swing_threshold) & 
             (self.data.model_data["weighted_prediction"] < lean_for_threshold), 'vote_prediction'] = 'lean_for'
        self.data.model_data.loc[(self.data.model_data["weighted_prediction"] >= lean_for_threshold) & 
             (self.data.model_data["weighted_prediction"] < swing_for_threshold), 'vote_prediction'] = 'swing_for'
        self.data.model_data.loc[self.data.model_data["weighted_prediction"] >= swing_for_threshold, 'vote_prediction'] = 'strongly_for'

        # Verify the distribution
        _vote_prediction = self.data.model_data['vote_prediction']
        print("\nLinear Model: Distribution of 7-level predictions:")
        print("=" * 50)
        print(_vote_prediction.value_counts(normalize=True).to_markdown(), end='\n\n')
        print("-" * 50)


        # Create a pie chart of the overall vote prediction counts
        _vote_counts = _vote_prediction.value_counts()
        plt.figure(figsize=(12, 12))
        _vote_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=True, figsize=(12, 12))
        plt.title('Linear Model: Overall Vote Prediction Distribution')
        plt.ylabel('')
        plt.savefig(FilePaths.IMAGE_PATH / 'vote_prediction_distribution.png')
        plt.show()

        # Print the counts
        print("\nLinear Model: Vote prediction counts:")
        print("=" * 50)
        print(_vote_counts.to_markdown())
        print("-" * 50)
        
        self.data.model_data['semi_generic_prediction'] = self.data.model_data['vote_prediction'].map({
            'strongly_for': 'for',
            'lean_for': 'for',
            'swing_for': 'swing',
            'swing_against': 'swing',
            'lean_against': 'against',
            'strongly_against': 'against'
        })
        self.data.model_data['generic_vote_prediction'] = self.data.model_data['vote_prediction'].map({
            'strongly_for': 'for',
            'lean_for': 'for',
            'swing_for': 'swing',
            'swing_against': 'swing',
            'lean_against': 'against',
            'strongly_against': 'against'
        })
        return self.data.model_data

