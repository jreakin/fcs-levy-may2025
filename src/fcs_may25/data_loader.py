from pathlib import Path
from typing import Union
import warnings

import polars as pl
from datetime import datetime
import numpy as np
import pandas as pd
from icecream import ic

warnings.filterwarnings('ignore')


# Setup
DOWNLOAD_PATH = Path.home() / 'Downloads'
VOTERFILE_PATH = Path.home() / 'PyCharmProjects' / 'state-voterfiles'
DATA_PATH = Path(__file__).parent / 'data'
PREDICTION_FOLDER = DATA_PATH / 'may25_predictions'

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
            bins=[18, 24, 34, 44, 54, 64, 74, 200],
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
        self.election_results = pd.read_csv(RESULTS).dropna(how='all', axis=1)
        self.election_results['for'] = pd.to_numeric(self.election_results['for'], errors='coerce').fillna(0)
        self.election_results['ward'] = self.election_results['precinct'].str[:-1]
        _election_cols = list(self.election_results.columns)
        self.election_results = self.election_results[[_election_cols.pop(_election_cols.index('ward'))] + _election_cols]
        return self


class ElectionReconstruction:
    data: FindlayVoterFile
    config: FindlayVoterFileConfig
    results_by_precinct: list[dict]
    age_pct: pd.DataFrame

    def __init__(self):
        self.data = FindlayVoterFile()
        self.config = self.data.config
        # self.estimate_by_precinct()
    # def estimate_by_precinct(self):
    #     # Estimate by precinct
    #     results_by_precinct = []
    #     _data = self.data.precinct_category_counts
    #     _categories = self.data._categories
    #     for precinct in _data.index:
    #         # Extract and ensure numeric
    #         X = _data.loc[precinct, _categories].values.reshape(1, -1)
    #         y = _data.loc[precinct, "for"]

    #         # Convert X to float explicitly
    #         X = np.array(X, dtype=float)
    #         y = float(y)  # Ensure y is a scalar float

    #         if len(results_by_precinct) < 3:  # Limit to first 3 precincts for debugging
    #             print(f"\nPrecinct: {precinct}")
    #             print(f"X dtype: {X.dtype}, X: {X}")
    #             print(f"y dtype: {type(y)}, y: {y}")

    #         total_voters = X.sum()
    #         if total_voters > 0 and y <= total_voters:
    #             result = lsq_linear(X, y, bounds=(0, 1))
    #             pi_c = result.x
    #         else:
    #             pi_c = [y / total_voters if total_voters > 0 else 0] * len(_categories)

    #         pi_dict = dict(zip(_categories, pi_c))
    #         precinct_estimates = [
    #             {"precinct": precinct, "category": cat, "pct": round(pct * 100, 2)}
    #             for cat, pct in pi_dict.items()
    #         ]
    #         results_by_precinct.extend(precinct_estimates)
    #     self.results_by_precinct = pd.DataFrame(results_by_precinct)
    #     self.results_by_precinct['ward'] = self.results_by_precinct['precinct'].str[:-1]
    #     self.results_by_precinct = self.results_by_precinct.sort_values(by=['ward','precinct', 'category'])
    #     return self
