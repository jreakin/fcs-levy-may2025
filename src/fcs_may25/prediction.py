import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from icecream import ic
import matplotlib.pyplot as plt
import warnings

import protocols as p_

from data_loader import (
    FilePaths, 
    FindlayVoterFileColumns as vf_cols, 
    NovemberResultsColumns as nov_results,
    FindlayVoterFileConfig as vf_config
)

def plot_regression_results(y_true, y_pred, model_name="Model"):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate residuals once
    residuals = y_true - y_pred
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.show()

    # 2. Residuals Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residuals Plot')
    plt.tight_layout()
    plt.show()

    # 3. Distribution of Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, density=True, alpha=0.7)
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.title(f'{model_name}: Distribution of Residuals')
    plt.tight_layout()
    plt.show()

    # Print regression metrics
    print(f"\nRegression Metrics:")
    print(f"R² Score: {r2_score(y_true, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.3f}")
    print(f"Mean Residual: {np.mean(residuals):.3f}")
    print(f"Std Residual: {np.std(residuals):.3f}")

# Example usage:
# plot_regression_results(y_test, model.predict(X_test), "Linear Regression")

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("Model doesn't support feature importance")
        return

    # Sort features by importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print(feature_importance.head(20).to_markdown(index=False))

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

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


# def add_total_for_vote_prediction(func):
#     def wrapper(df, config, *args, **kwargs):
#         results = func(df, config, *args, **kwargs)
        
#         # Initialize missing columns with zeros if they don't exist
#         for col in ['strongly_for', 'lean_for', 'strongly_against', 'lean_against', 'swing_against', 'swing', 'swing_for']:
#             if col not in results.columns:
#                 results[col] = 0
        
#         # Calculate shares using ward-level totals
#         ward_totals = results['Total']
#         results['total_for_share'] = (results['strongly_for'] + results['lean_for'])
#         results['total_against_share'] = ((results['strongly_against'] + results['lean_against']))
#         results['total_swing_share'] = ((results['swing_against'] + results['swing'] + results['swing_for']))
        
#         # Add total row using overall totals
#         total_voters = results['Total'].sum()
#         results.loc['Total', 'total_for_share'] = ((results['strongly_for'].sum() + results['lean_for'].sum()))
#         results.loc['Total', 'total_against_share'] = ((results['strongly_against'].sum() + results['lean_against'].sum()))
#         results.loc['Total', 'total_swing_share'] = (results['swing_against'].sum() + results['swing'].sum() + results['swing_for'].sum())
        
#         return results
#     return wrapper

# @add_total_for_vote_prediction
# def vote_prediction_by_age(df, config):
#     return pd.crosstab(
#         index=df['AGE_RANGE'],
#         columns=df['vote_prediction'],
#         margins=True,
#         margins_name="Total",
#     )

# @add_total_for_vote_prediction
# def vote_prediction_by_ward(df, config):
#     return pd.crosstab(
#         index=df['WARD'],
#         columns=df['vote_prediction'],
#         margins=True,
#         margins_name="Total",
#     )

# @add_total_for_vote_prediction
# def vote_prediction_by_precinct(df, config):
#     return pd.crosstab(
#         index=df['PRECINCT_NAME'],
#         columns=df['vote_prediction'],
#         margins=True,
#         margins_name="Total",
#     )

# def groupby_ward_vote_share(df):
#     return df.reset_index().groupby('WARD').agg({
#         'total_for_count': 'sum',
#         'total_against_count': 'sum',
#         'total_swing_count': 'sum',
#     }).reset_index()

# def groupby_age_vote_share(df: pd.DataFrame) -> pd.DataFrame:
#     return df.reset_index().groupby('AGE_RANGE').agg({
#         'total_for_count': 'sum',
#         'total_against_count': 'sum',
#         'total_swing_count': 'sum',
#     }).reset_index()

# def plot_vote_share(df: pd.DataFrame, by: str, cols: list[str],  title: str):
#     plot_df = df.reset_index().drop(columns=['Total'])
#     plot_df = plot_df.iloc[:-1]
#     ic(plot_df.columns.to_list())
#     plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
#     plt.savefig(FilePaths.IMAGE_PATH / f'{title}.png')
#     plt.show()

# # def plot_vote_share_by_level(df, by: str, cols: list[str], title: str):
# #     plot_df = df.reset_index()
# #     ic(plot_df.columns.to_list())
# #     plot_df.plot.bar(stacked=False, x=by, y=cols, title=title, figsize=(16, 12))
# #     plt.show()
# def plot_pie_chart(df: pd.DataFrame, title: str):
#     df.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, figsize=(8, 8), ylabel='', title=title)
#     plt.title(title)
#     plt.savefig(FilePaths.IMAGE_PATH / f'{title}.png')
#     plt.show()
    

class FindlayPredictionModel:
    model_data: pd.DataFrame
    ward_mapping: dict[str, float]
    precinct_mapping: dict[str, float]
    preprocessor: ColumnTransformer
    X_pseudo: pd.DataFrame
    y_pseudo: pd.Series
    model: LogisticRegression
    feature_names: list[str]
    feature_importance: pd.DataFrame

    def __init__(self, data: p_.ModelDataStartingPoint):
        self.model_data = data.model_data
        self.ward_mapping = data.model_data.groupby(vf_cols.WARD)[nov_results.FOR_SHARE].mean().to_dict()
        self.precinct_mapping = data.model_data.groupby(vf_cols.PRECINCT_NAME)[nov_results.FOR_SHARE].mean().to_dict()



    
    # def add_feature_columns(self, category_cols: p_.LinearModelFeatureColumns = ml_cat, voterfile_cols: p_.VoterFileColumns = vf_cols):
    #     _model_data = self.data.model_data
    #     _model_data[category_cols.AGE_RANGE_CAT] = pd.Categorical(_model_data[vf_cols.AGE_RANGE], categories=vf_config.AGE_RANGE_SORTED, ordered=True)
    #     _model_data[category_cols.PARTY_CAT] = pd.Categorical(_model_data[vf_cols.PARTY_AFFILIATION], categories=['D', 'I', 'R'], ordered=True)
    #     _model_data[category_cols.AGE_WARD] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.WARD].astype(str)
    #     _model_data[category_cols.AGE_PRECINCT] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.PRECINCT_NAME].astype(str)
    #     _model_data[category_cols.AGE_PARTY] = _model_data[voterfile_cols.AGE_RANGE].astype(str) + '-' + _model_data[voterfile_cols.PARTY_AFFILIATION].astype(str)
    #     _model_data['P_SCORE_LAST4_CAT'] = pd.cut(
    #         _model_data[ml_cat.P_SCORE],
    #         bins=5,
    #         labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    #     ).astype(int)
    #     _model_data['G_SCORE_LAST4_CAT'] = pd.cut(
    #         _model_data[ml_cat.G_SCORE],
    #         bins=5,
    #         labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    #     ).astype(int)
    #     _model_data['P_SCORE_ALL_CAT'] = pd.cut(
    #         _model_data['P_SCORE_ALL'],
    #         bins=5,
    #         labels=[0, 2, 4, 6, 8] # strongly against, lean against, lean for, strongly for
    #     ).astype(int)
    #     _model_data['G_SCORE_ALL_CAT'] = pd.cut(
    #         _model_data['G_SCORE_ALL'],
    #         bins=5,
    #         labels=[0, 2, 4, 6, 8] # strongly against, lean against, lean for, strongly for
    #     ).astype(int)
    #     _model_data['P_SCORE_LAST4_AGE_WARD_PRECINCT'] = _model_data['P_SCORE_LAST4_CAT'].astype(str) + '-' + _model_data[category_cols.AGE_WARD]
    #     _model_data['G_SCORE_LAST4_AGE_WARD_PRECINCT'] = _model_data['G_SCORE_LAST4_CAT'].astype(str) + '-' + _model_data[category_cols.AGE_WARD]
    #     _model_data['P_SCORE_ALL_AGE_WARD_PRECINCT'] = _model_data['P_SCORE_ALL_CAT'].astype(str) + '-' + _model_data[category_cols.AGE_WARD]
    #     _model_data['G_SCORE_ALL_AGE_WARD_PRECINCT'] = _model_data['G_SCORE_ALL_CAT'].astype(str) + '-' + _model_data[category_cols.AGE_WARD]
    #     ml_cat.interaction_features.extend([
    #         'P_SCORE_LAST4_AGE_WARD_PRECINCT',
    #         'G_SCORE_LAST4_AGE_WARD_PRECINCT',
    #         'P_SCORE_ALL_AGE_WARD_PRECINCT',
    #         'G_SCORE_ALL_AGE_WARD_PRECINCT',
    #     ])
    #     self.data.model_data = _model_data
    #     return self
    
    # def create_preprocessor(self, feature_lists: p_.LinearModelFeatureLists = ml_cat):
    #     self.preprocessor = ColumnTransformer(
    #         transformers=[
    #             ("cat", OneHotEncoder(drop="first", sparse_output=False), feature_lists.category_features),
    #             ("high_card", TargetEncoder(), feature_lists.high_cardinality_features),
    #             ("num", StandardScaler(), feature_lists.numerical_features),
    #             ("interaction", TargetEncoder(), feature_lists.interaction_features),
    #         ]
    #     )
    #     return self

    # def fit_pseudo_preprocessor(self) -> tuple[pd.DataFrame, pd.Series]:
    #     self.X_pseudo = self.preprocessor.fit_transform(self.data.model_data, self.data.model_data[nov_results.FOR_SHARE])
    #     # More granular thresholds based on your data distribution
    #     self.y_pseudo = pd.cut(
    #         self.data.model_data[nov_results.FOR_SHARE],
    #         bins=[-np.inf, 0.15, 0.35, 0.45, 0.55, 0.65, 0.85, np.inf],
    #         labels=[0, 1, 2, 3, 4, 5, 6]  # strongly_against to strongly_for
    #     ).astype(int)
    #     return self.X_pseudo, self.y_pseudo
    
    # def run(self):
    #     self.add_feature_columns()
    #     self.create_preprocessor()
    #     self.fit_pseudo_preprocessor()
    #     self.run_max_iter_test()
    #     self.fit_model()
    #     self.get_feature_names()
    #     self.predict_probability()
    #     self.create_weighted_prediction()
    #     return self
    
    # def run_max_iter_test(self):
    #     X, y = self.fit_pseudo_preprocessor()
    #     return test_max_iter_impact(X, y)
    
    # def fit_model(self):
    #     self.model = LogisticRegression(penalty=None, random_state=42, max_iter=500, multi_class='multinomial')
    #     self.model.fit(self.X_pseudo, self.y_pseudo)
    #     return self.model
    
    # def get_feature_names(self):
    #     self.feature_names = self.preprocessor.get_feature_names_out()
    #     self.feature_importance = pd.DataFrame({
    #         'feature': self.feature_names,
    #         'against': np.abs(self.model.coef_[0]),
    #         'swing': np.abs(self.model.coef_[1]),
    #         'for': np.abs(self.model.coef_[2])
    #     })
    #     # Calculate overall importance as the mean across all classes
    #     self.feature_importance['overall_importance'] = self.feature_importance[['against', 'swing', 'for']].mean(axis=1)

    #     # Sort by overall importance
    #     self.feature_importance = self.feature_importance.sort_values('overall_importance', ascending=False)

    #     # Display the top 15 most important features
    #     print("Linear Model: Top 15 most important features (overall):")
    #     print("=" * 50)
    #     print(self.feature_importance[['feature', 'overall_importance']].head(15).to_markdown(index=False))
    #     print("-" * 50)

    #     # Visualize the top 15 features
    #     plt.figure(figsize=(12, 8))
    #     plt.barh(self.feature_importance['feature'][:15], self.feature_importance['overall_importance'][:15])
    #     plt.xlabel('Linear Model: Feature Importance (Absolute Coefficient Value)')
    #     plt.ylabel('Feature')
    #     plt.title('Linear Model: Top 15 Most Important Features')
    #     plt.tight_layout()
    #     plt.savefig(FilePaths.IMAGE_PATH / 'linear_model_feature_importance.png')
    #     plt.close()

    #     return self.feature_names
    
    # def predict_probability(self):
    #     """Predict probabilities for each class."""
    #     # Get predictions directly
    #     predictions = self.model.predict(self.X_pseudo)
        
    #     # Map predictions to probabilities, shifted up to center around 0.5
    #     prob_map = {
    #         0: 0.47,  # strongly_against 
    #         1: 0.48,  # lean_against
    #         2: 0.49,  # swing_against
    #         3: 0.50,  # swing
    #         4: 0.51,  # swing_for
    #         5: 0.52,  # lean_for
    #         6: 0.53   # strongly_for
    #     }
        
    #     self.data.model_data['P_for'] = pd.Series(predictions).map(prob_map)
    #     return self.data.model_data
    
    # def create_weighted_prediction(self, precinct: str = None, ward: str = None):
    #     """Create weighted predictions based on November results and participation."""
    #     # Calculate participation scores
    #     primary_participation = self.data.model_data[VoterScoringColumns.PRIMARY_SCORE] / 4.0
    #     general_participation = self.data.model_data[VoterScoringColumns.GENERAL_SCORE] / 4.0
        
    #     # Get ward-level results and calculate swing band
    #     ward_results = self.ward_mapping
        
    #     # For each ward, calculate the swing band (±7.5% around the ward's result)
    #     ward_swing_bands = {}
    #     for ward, result in ward_results.items():
    #         ward_swing_bands[ward] = {
    #             'lower': result - 0.075,  # 7.5% below ward result
    #             'upper': result + 0.075   # 7.5% above ward result
    #         }
        
    #     # Create base prediction using ward results and participation
    #     self.data.model_data['weighted_prediction'] = (
    #         0.6 * self.data.model_data[vf_cols.WARD].map(ward_results) +  # Ward November results (60%)
    #         0.25 * primary_participation +  # Primary participation (25%)
    #         0.15 * general_participation    # General participation (15%)
    #     )

    #     # For each ward, determine the thresholds based on that ward's November results
    #     def assign_category(row):
    #         ward = row[vf_cols.WARD]
    #         pred = row['weighted_prediction']
            
    #         # Get swing band for this ward
    #         swing_band = ward_swing_bands[ward]
            
    #         # Determine category based on ward-specific thresholds
    #         if pred < swing_band['lower'] - 0.05:
    #             return FindlayPredictionGranularTiers.STRONGLY_AGAINST
    #         elif pred < swing_band['lower']:
    #             return FindlayPredictionGranularTiers.LEAN_AGAINST
    #         elif pred < swing_band['lower'] + 0.025:
    #             return FindlayPredictionGranularTiers.SWING_AGAINST
    #         elif pred <= swing_band['upper'] - 0.025:
    #             return FindlayPredictionGranularTiers.SWING
    #         elif pred <= swing_band['upper']:
    #             return FindlayPredictionGranularTiers.SWING_FOR
    #         elif pred <= swing_band['upper'] + 0.05:
    #             return FindlayPredictionGranularTiers.LEAN_FOR
    #         else:
    #             return FindlayPredictionGranularTiers.STRONGLY_FOR
        
    #     # Apply the ward-specific categorization
    #     self.data.model_data['vote_prediction'] = self.data.model_data.apply(assign_category, axis=1)
        
    #     # Log the distribution by ward for verification
    #     for ward in ward_results.keys():
    #         ward_dist = self.data.model_data[self.data.model_data[vf_cols.WARD] == ward]['vote_prediction'].value_counts()
    #         ic(f"Ward {ward} distribution:\n{ward_dist}")
        
    #     return self.data.model_data
