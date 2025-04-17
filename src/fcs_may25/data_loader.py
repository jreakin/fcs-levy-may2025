from pathlib import Path
from typing import Union
import warnings

import polars as pl
from datetime import datetime
import numpy as np
import pandas as pd
from icecream import ic
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

warnings.filterwarnings('ignore')


# Setup
DOWNLOAD_PATH = Path.home() / 'Downloads'
VOTERFILE_PATH = Path.home() / 'PyCharmProjects' / 'state-voterfiles'
DATA_PATH = Path(__file__).parent / 'data'
PREDICTION_FOLDER = DATA_PATH / 'may25_predictions'
IMAGE_PATH = DATA_PATH / 'images'

RESULTS = DATA_PATH / 'NOV24-FCS-TAX.csv'
EARLY_VOTE = DATA_PATH / 'may26_ev'
DATA = VOTERFILE_PATH / "data/ohio/voterfile/ohio-statewide"

DataFrameType = Union[pl.LazyFrame, pd.DataFrame]

class FindlayVoterFileConfig:
    levy_election_date = datetime.strptime("2025-05-06", "%Y-%m-%d").date()
    PARTY_WEIGHTS = {
        'D': 1.5,  # Democrats weighted highest
        'R': 1.0,  # Republicans weighted higher than Independents
        'I': 1.2   # Independents as baseline
    }
    BASE_COLUMNS = [
        "SOS_VOTERID", 
        "PRECINCT_NAME", 
        "PARTY_AFFILIATION", 
        "WARD", "AGE",
        "AGE_RANGE", 
        "CATEGORY", 
        "WEIGHT",
        'VOTED_MAY_LEVY', 
        'VOTED_IN_NOV', 
        'NO_IN_NOV_YES_IN_MAY'
    ]
    PRIMARY_COLUMNS = {}
    GENERAL_COLUMNS = {}
    ELECTION_DATES = {}
    ELECTION_COLUMNS = []

class FindlayVoterFile:
    data: pl.DataFrame
    config: FindlayVoterFileConfig
    model_data: pd.DataFrame
    weighted_data: pd.DataFrame
    primary_scores: pd.DataFrame
    weighted_precinct_category_counts: pd.DataFrame
    turnout_data: pd.DataFrame
    current_votes: pd.DataFrame
    election_results: pd.DataFrame
    _categories: list[str]
    _last4_primaries: list[str]
    _last4_generals: list[str]

    def __init__(self):
        self.config = FindlayVoterFileConfig()
        self.load_early_votes()
        self.load_data()
        self.load_model_data()
        self.load_weighted_data()
        self.aggregate_data()
        self.include_election_results()

    def load_early_votes(self):
        vote_lists = []
        for f in EARLY_VOTE.glob("*.csv"):
            f = Path(f)
            df = pd.read_csv(f)
            df['DATE ENTERED:'] = pd.to_datetime(df['DATE ENTERED:'])
            if 'DATE RETURNED:' in df.columns:
                df['DATE RETURNED:'] = pd.to_datetime(df['DATE RETURNED:'])
            ic(f.stem)
            if 'In Office' in f.stem:
                df['Vote Method'] = 'In-Person'
            elif 'By Mail' in f.stem:
                df['Vote Method'] = 'Mail'
            vote_lists.append(df)
        self.current_votes = pd.concat(vote_lists).drop_duplicates(subset=['STATE ID#'])

    def load_data(self):
        _data = pl.scan_csv(list(DATA.glob("*.txt")), separator=',', encoding='utf8-lossy')
        _data = _data.filter(
            (pl.col("CITY_SCHOOL_DISTRICT") == "FINDLAY CITY SD")
            & (pl.col("COUNTY_NUMBER") == 32)
        )
        _data = _data.with_columns(
            pl.col("DATE_OF_BIRTH").str.strptime(pl.Datetime, format="%Y-%m-%d"),
            pl.col("REGISTRATION_DATE").str.strptime(pl.Datetime, format="%Y-%m-%d"),
        )
        _data = _data.with_columns(
            # Calculate age in years
            (
            (pl.lit(datetime.now()) - pl.col("DATE_OF_BIRTH"))
            .dt.total_days()
            // 365  # Integer division to get approximate years
        ).cast(pl.Int64).alias("AGE")
                )
        data = _data.collect().to_pandas()
        data['PARTY_AFFILIATION'] = data['PARTY_AFFILIATION'].where(data['PARTY_AFFILIATION'] != '', 'I')
        data["AGE_RANGE"] = pd.cut(
            data["AGE"],
            bins=[17, 24, 34, 44, 54, 64, 74, 200],
            labels=[
                "18-24",
                "25-34", 
                "35-44",
                "45-54",
                "55-64",
                "65-74",
                "75+"
            ],
        )
        data['WARD'] = data['PRECINCT_NAME'].str[:-1]
        data['CATEGORY'] = data['AGE_RANGE'].astype(str) + '-' + data['PARTY_AFFILIATION'].astype(str)
        self._categories = data['CATEGORY'].unique().tolist()
        data['WEIGHT'] = data['PARTY_AFFILIATION'].map(self.config.PARTY_WEIGHTS).fillna(1.0)
        self.config.ELECTION_DATES = {
                x: datetime.strptime(x.split("-")[1], "%m/%d/%Y").date()
                    for x in data.columns if "GENERAL" in x or "PRIMARY" in x
            }
        
        self.config.PRIMARY_COLUMNS = {k: v for k, v in self.config.ELECTION_DATES.items() if "PRIMARY" in k}
        self.config.GENERAL_COLUMNS = {k: v for k, v in self.config.ELECTION_DATES.items() if "GENERAL" in k}
        self._last4_primaries = list(self.config.PRIMARY_COLUMNS.keys())[-4:]
        self._last4_generals = list(self.config.GENERAL_COLUMNS.keys())[-4:]
        self.config.ELECTION_COLUMNS = self._last4_primaries + self._last4_generals
        data = data.dropna(axis=1, how="all")
        self.data = data
        self.data['VOTED_MAY_LEVY'] = self.data['SOS_VOTERID'].isin(self.current_votes['STATE ID#'])
        self.data['VOTED_IN_NOV'] = self.data[list(self.config.GENERAL_COLUMNS.keys())[-1]].astype(bool)
        self.data['NO_IN_NOV_YES_IN_MAY'] = ~self.data['VOTED_IN_NOV'] & self.data['VOTED_MAY_LEVY']

    
    def load_model_data(self):
        model_data: pd.DataFrame = self.data[self.config.BASE_COLUMNS + self.config.ELECTION_COLUMNS]
        model_data[self.config.ELECTION_COLUMNS] = model_data[self.config.ELECTION_COLUMNS].map(lambda x: 1 if x else 0)
        model_data['P_SCORE'] = model_data[self._last4_primaries].sum(axis=1).round(2)
        model_data['G_SCORE'] = model_data[self._last4_generals].sum(axis=1).round(2)
        model_data['PERCENT_SCORE'] = model_data[self._last4_primaries + self._last4_generals].sum(axis=1).round(2)
        self.model_data = model_data
    
    def load_weighted_data(self):
        weighted_data = self.model_data.copy()
        weighted_data['P_SCORE'] = (weighted_data['P_SCORE'] * weighted_data['WEIGHT']).round(2)
        weighted_data['G_SCORE'] = (weighted_data['G_SCORE'] * weighted_data['WEIGHT']).round(2)
        weighted_data['PERCENT_SCORE'] = (weighted_data['PERCENT_SCORE'] * weighted_data['WEIGHT']).round(2)
        self.weighted_precinct_category_counts = weighted_data.groupby(['PRECINCT_NAME', 'CATEGORY'])['WEIGHT'].sum().round(2).unstack(fill_value=0)
        self.weighted_data = weighted_data

    def aggregate_data(self):
        self.primary_scores = self.weighted_data.groupby('CATEGORY').agg(
            {
                'G_SCORE': 'mean', 
                'P_SCORE': 'mean',
                'PERCENT_SCORE': 'mean'
            }
        ).round(2).reset_index()
        self.turnout_data = self.model_data[self.model_data[list(self.config.GENERAL_COLUMNS.keys())[-1]] > 0]
        self.turnout_data['ward'] = self.turnout_data['PRECINCT_NAME'].str[:-1]
        return self
    
    def include_election_results(self):
        election_results = pd.read_csv(RESULTS).dropna(how='all', axis=1)
        election_results['for'] = pd.to_numeric(election_results['for'], errors='coerce').fillna(0)
        election_results['ward'] = election_results['precinct'].str[:-1]
        _election_cols = list(election_results.columns)
        election_results = election_results[[_election_cols.pop(_election_cols.index('ward'))] + _election_cols]
        election_results['nov_for_pct'] = (election_results['for'] / election_results['total']).round(4)
        election_results['nov_against_pct'] = (election_results['against'] / election_results['total']).round(4)
        self.election_results = election_results
        return self


class ElectionReconstruction:
    data: FindlayVoterFile
    config: FindlayVoterFileConfig
    results_by_precinct: list[dict]
    age_pct: pd.DataFrame
    november_counts: pd.DataFrame = None
    november_total_voters: pd.DataFrame = None
    november_proportions: pd.DataFrame = None
    november_proportions_cols: list[str] = None
    november_turnout: pd.DataFrame
    election_results: pd.DataFrame
    prepared_election_results: pd.DataFrame = None
    ecological_results: dict = None
    comparison_results: dict = None
    may_prediction_votes: pd.DataFrame = None
    prediction_dummies: pd.DataFrame = None
    prediction_by_age_range: pd.DataFrame = None
    prediction_by_ward: pd.DataFrame = None
    prediction_by_precinct: pd.DataFrame = None
    prediction_and_results: pd.DataFrame = None
    def __init__(self):
        self.data = FindlayVoterFile()
        self.config = self.data.config
        self.november_turnout = self.data.turnout_data
        self.election_results = self.data.election_results
        # self.estimate_by_precinct()


    def run(self):
        self.get_november_turnout_counts()
        self.get_november_turnout_proportions()
        self.get_november_age_pivot()
        self.prepare_election_results()
        self.build_ecological_regression()
        self.build_comparison_model1()
        self.build_logistic_regression()
        self.build_comparison_model2()
        self.calculate_percentages()
        return self

    def get_november_turnout_counts(self):
        self.november_counts = self.november_turnout.groupby(['PRECINCT_NAME', 'AGE_RANGE']).size().reset_index(name='count')
        self.november_total_voters = self.november_turnout.groupby('PRECINCT_NAME').size().reset_index(name='total')
        return self
    
    def get_november_turnout_proportions(self):
        self.november_proportions = self.november_counts.merge(self.november_total_voters, on='PRECINCT_NAME')
        self.november_proportions['prop_age'] = self.november_proportions['count'] / self.november_proportions['total']
        return self
    
    def get_november_age_pivot(self):
        self.november_age_pivot = self.november_proportions.pivot(index='PRECINCT_NAME', columns='AGE_RANGE', values='prop_age').fillna(0).reset_index()
        self.november_proportions_cols = self.november_age_pivot.columns.tolist()
        self.november_proportions_cols.pop(self.november_proportions_cols.index('PRECINCT_NAME'))
        return self
    
    def prepare_election_results(self):
        self.election_results = self.election_results.rename(columns={"precinct": "PRECINCT_NAME"})
        data = self.november_age_pivot.merge(self.election_results, left_on="PRECINCT_NAME", right_on="PRECINCT_NAME")
        data['p_for'] = data['for'] / data['total']
        self.prepared_election_results = data
        return self

    def build_ecological_regression(self):
        eco_results = {}
        # Select numeric age range columns for X
        X = self.prepared_election_results[self.november_proportions_cols]
        y = self.prepared_election_results['p_for'].astype(float)

        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # Add age information - modified to work with the actual data structure
        age_ranges = ['0-18', '18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+']

        # Create a mapping from age range to rank
        age_rank_map = {age: i for i, age in enumerate(age_ranges)}

        # Create a mapping from age range to rank
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
        y = self.prepared_election_results['p_for'].astype(float)

        # Fit constrained least squares model (beta coefficients between 0 and 1)
        eco_results['result'] = lsq_linear(X_np, y_np, bounds=(0, 1))
        eco_results['beta_k_constrained'] = eco_results['result'].x

        # Map coefficients to age groups
        eco_results['age_groups'] = sorted(self.november_turnout['AGE_RANGE'].dropna().astype(str).unique().tolist())
        eco_results['beta_k_constrained_series'] = pd.Series(eco_results['beta_k_constrained'], index=eco_results['age_groups'])
        eco_results['X_np'] = X_np
        eco_results['y_np'] = y_np
        eco_results['X'] = X
        eco_results['y'] = y
        self.ecological_results = eco_results

        ic("Constrained Beta Coefficients:")
        ic(eco_results['beta_k_constrained_series'])
        return self
    
    def build_comparison_model1(self):
        comparison_results = {}
        # Fit OLS model to estimate initial group probabilities
        model = sm.OLS(self.ecological_results['y_np'], self.ecological_results['X_np']).fit()
        beta_k = model.params

        predicted_p_for = model.fittedvalues
        actual_p_for = self.ecological_results['y']

        r_squared = model.rsquared
        ic(f"R-squared: {r_squared:.4f}")

        mae = np.mean(np.abs(actual_p_for - predicted_p_for))
        ic(f"Mean Absolute Error: {mae:.4f}")

        rmse = np.sqrt(np.mean((actual_p_for - predicted_p_for)**2))
        ic(f"Root Mean Squared Error: {rmse:.4f}")

        predicted_for_votes = predicted_p_for * self.prepared_election_results['total']
        actual_for_votes = self.prepared_election_results['for']
        mae_votes = np.mean(np.abs(actual_for_votes - predicted_for_votes))
        ic(f"MAE for 'for' votes: {mae_votes:.2f}")
        ic("Beta coefficients:")
        ic(beta_k)

        comparison_results['predicted_p_for'] = predicted_p_for
        comparison_results['actual_p_for'] = actual_p_for
        comparison_results['r_squared'] = r_squared
        comparison_results['mae'] = mae
        comparison_results['rmse'] = rmse
        comparison_results['mae_votes'] = mae_votes
        comparison_results['beta_k'] = beta_k
        self.comparison_results = comparison_results
        if any(beta_k < 0) or any(beta_k > 1):
            ic("Note: Some beta coefficients are outside [0,1]")
        return self
    
    
    def build_logistic_regression(self):
        # Step 1: Clip beta_k to [0,1] to ensure valid probabilities
        beta_k_clipped = np.clip(self.comparison_results['beta_k'], 0, 1)

        # Create a dictionary mapping age ranges to their corresponding beta values
        age_beta_map = {age: beta for age, beta in zip(self.ecological_results['age_groups'], beta_k_clipped)}

        # Step 2: Assign initial probabilities to voters based on age group
        self.november_turnout['prob_for'] = self.november_turnout['AGE_RANGE'].map(age_beta_map)

        # Ensure all probabilities are valid (between 0 and 1) and not NaN
        self.november_turnout['prob_for'] = self.november_turnout['prob_for'].fillna(0)  # Replace NaN with 0.5
        self.november_turnout['prob_for'] = self.november_turnout['prob_for'].clip(0, 1)  # Clip to [0, 1]

        # Step 3: Calculate predicted votes
        np.random.seed(42)  # For reproducibility
        self.november_turnout['simulated_vote'] = np.random.binomial(1, self.november_turnout['prob_for'])

        X_features = pd.get_dummies(self.november_turnout[['AGE_RANGE', 'PRECINCT_NAME']], drop_first=True)
        y_target = self.november_turnout['simulated_vote']

        X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
    
        y_pred = log_reg.predict(X_test)
        ic("\nLogistic Regression Performance:")
        ic(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        ic("Classification Report:")
        print(classification_report(y_test, y_pred))

        self.november_turnout['predicted_prob_for_logreg'] = log_reg.predict_proba(X_features)[:, 1]

        # Step 9: Assign vote predictions based on logistic regression probabilities
        self.november_turnout['vote_prediction_logreg'] = np.where(
            self.november_turnout['predicted_prob_for_logreg'] < 0.5, 'against',
            np.where(self.november_turnout['predicted_prob_for_logreg'] > 0.65, 'for', 'swing')
        )
        
        return self
    
    def build_comparison_model2(self):
        # --- Comparison of Predictions ---
        # Compare aggregated logistic regression predictions to actual precinct-level p_for

        predicted_p_for_logreg = self.november_turnout.groupby('PRECINCT_NAME')['predicted_prob_for_logreg'].mean()
        comparison = pd.DataFrame({
            'actual_p_for': self.prepared_election_results.set_index('PRECINCT_NAME')['p_for'],
            'predicted_p_for_logreg': predicted_p_for_logreg
        })
        ic("\nComparison of Aggregated Predictions vs. Actual Precinct Proportions:")
        ic(comparison)
        # --- Original OLS-Based Predictions ---
        # Retain OLS-based predictions for comparison
        # Create a dictionary mapping age ranges to their corresponding beta values

        beta_k_dict = {age: beta for age, beta in zip(self.ecological_results['age_groups'], self.comparison_results['beta_k'])}
        self.november_turnout['predicted_prob_for'] = self.november_turnout['AGE_RANGE'].map(beta_k_dict).astype(float)
        self.november_turnout['vote_prediction'] = pd.cut(
            self.november_turnout['predicted_prob_for'],
            bins=[-np.inf, 0.45, 0.55, np.inf],
            labels=['against', 'swing', 'for']
        )
        prediction_dummies = pd.get_dummies(
            self.november_turnout['vote_prediction'],
            prefix='prediction',
            columns=['against', 'swing', 'for']
        )
        
        # Ensure all necessary columns exist
        for col in ['prediction_against', 'prediction_swing', 'prediction_for']:
            if col not in prediction_dummies.columns:
                prediction_dummies[col] = 0
            
        self.november_turnout[list(prediction_dummies.columns)] = prediction_dummies

        # --- Calculate Results ---
        prediction_by_precinct = self.november_turnout.groupby(['ward', 'PRECINCT_NAME'])[list(prediction_dummies.columns)].sum().reset_index()
        results_and_prediction = self.election_results.merge(prediction_by_precinct, left_on='PRECINCT_NAME', right_on='PRECINCT_NAME')
        results_and_prediction['for_diff'] = results_and_prediction['for'] - results_and_prediction['prediction_for']
        results_and_prediction['against_diff'] = results_and_prediction['against'] - results_and_prediction['prediction_against']
        result_agg_cols = {x: 'sum' for x in list(prediction_dummies.columns)}
        result_agg_cols['SOS_VOTERID'] = 'count'
        may_prediction_votes = (self.november_turnout[self.november_turnout['VOTED_MAY_LEVY'] == 1]
                                .groupby(['ward', 'PRECINCT_NAME', 'AGE_RANGE'])
                                .agg(result_agg_cols)
                                .rename(columns={'SOS_VOTERID': 'total_votes'}))
        self.may_prediction_votes = may_prediction_votes
        self.prediction_dummies = prediction_dummies
        return self

    def calculate_percentages(self):
        # Ensure all necessary columns exist in the prediction data
        required_columns = ['prediction_for', 'prediction_against', 'prediction_swing']
        
        prediction_by_precinct = self.may_prediction_votes.groupby('PRECINCT_NAME')[
            ['total_votes'] + list(self.prediction_dummies.columns)
        ].sum().reset_index()
        
        # Calculate percentages only if columns exist
        total_votes = prediction_by_precinct['total_votes']
        
        for col in required_columns:
            if col in prediction_by_precinct.columns:
                prediction_by_precinct[f'pct_{col.split("_")[1]}'] = (
                    prediction_by_precinct[col] / total_votes
                ).round(2)
            else:
                prediction_by_precinct[col] = 0
                prediction_by_precinct[f'pct_{col.split("_")[1]}'] = 0
        
        prediction_and_results = prediction_by_precinct.merge(self.election_results, left_on='PRECINCT_NAME', right_on='PRECINCT_NAME')
        prediction_and_results['better_than_nov'] = prediction_and_results['pct_for'] > prediction_and_results['nov_for_pct']
        prediction_and_results['winning'] = prediction_and_results['prediction_for'] >= prediction_and_results['prediction_against']

        prediction_by_ward = self.may_prediction_votes.groupby('ward')[['total_votes'] + list(self.prediction_dummies.columns)].sum().reset_index()
        prediction_by_ward['pct_for'] = (prediction_by_ward['prediction_for'] / prediction_by_ward['total_votes']).round(2)
        prediction_by_ward['pct_against'] = (prediction_by_ward['prediction_against'] / prediction_by_ward['total_votes']).round(2)
        prediction_by_ward['pct_swing'] = (prediction_by_ward['prediction_swing'] / prediction_by_ward['total_votes']).round(2)
        prediction_by_ward['ward_pct_of_vote'] = (prediction_by_ward['total_votes'] / prediction_by_ward['total_votes'].sum()).round(2)

        prediction_by_age_range = self.may_prediction_votes.groupby('AGE_RANGE')[list(self.prediction_dummies.columns)].sum().reset_index()
        prediction_by_age_range['total_votes'] = prediction_by_age_range['prediction_for'] + prediction_by_age_range['prediction_against'] + prediction_by_age_range['prediction_swing']
        prediction_by_age_range['pct_for'] = (prediction_by_age_range['prediction_for'] / prediction_by_age_range['total_votes']).round(2)
        prediction_by_age_range['pct_against'] = (prediction_by_age_range['prediction_against'] / prediction_by_age_range['total_votes']).round(2)
        self.prediction_by_age_range = prediction_by_age_range
        self.prediction_by_ward = prediction_by_ward
        self.prediction_by_precinct = prediction_by_precinct
        self.prediction_and_results = prediction_and_results
        return self
# Assess OLS model accuracy
