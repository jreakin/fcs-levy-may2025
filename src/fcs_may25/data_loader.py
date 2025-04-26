from __future__ import annotations
from dataclasses import dataclass
from protocols import ModelColumnSetup, ModelConfig, VotingMethod, VoterScoringColumns
from config import (
    FindlayModelConfig, 
    FilePaths, 
    FindlayVoterFileColumns, 
    FindlayEarlyVoteColumns, 
    FindlayPredictionGranularTiers, 
    FindlayPredictionTotalTiers,
    NovemberResultsColumns,
    FindlayMLModelCategories
)
from pathlib import Path
from typing import ClassVar, Optional
import warnings
from contextlib import ExitStack

import polars as pl
from datetime import datetime
import numpy as np
import pandas as pd
from icecream import ic
import functools

warnings.filterwarnings('ignore')


class DataWrapper:

    @staticmethod
    def format_date(*date_columns, fmt: Optional[str] = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                df = func(self, *args, **kwargs)
                ic("format_date", date_columns)
                for col in date_columns:
                    if isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame):
                        # For Polars, we'll add the column transformation directly
                        df = df.with_columns(
                            pl.col(col).str.strptime(pl.Datetime, format=fmt)
                        )
                    else:
                        # For Pandas
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], format=fmt)
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def create_age_column(date_of_birth_col: str, age_col_name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                df = func(self, *args, **kwargs)
                if isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame):
                    df = df.with_columns(
                        # Calculate age in years
                        (
                        (pl.lit(datetime.now()) - pl.col(date_of_birth_col))
                            .dt.total_days()
                            // 365  # Integer division to get approximate years
                        )
                        .cast(pl.Int64)
                        .alias(age_col_name)
                    )
                else:
                    df[age_col_name] = (datetime.now() - df[date_of_birth_col]).dt.days // 365
                return df
            return wrapper
        return decorator
    

    @staticmethod
    def create_age_range(age_col: str, age_range_col: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                df = func(self, *args, **kwargs)
                if isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame):
                    df = df.with_columns(
                        pl.col(age_col).cast(pl.Int64).alias(age_col)
                    )
                    # Bin the age to age_range
                    df = df.with_columns(
                        pl.when(pl.col(age_col) < 18)
                            .then(pl.lit('18-24'))
                            .when(pl.col(age_col) >= 18)
                            .then(pl.lit('25-34'))
                            .when(pl.col(age_col) >= 34)
                            .then(pl.lit('35-44'))
                            .when(pl.col(age_col) >= 44)
                            .then(pl.lit('45-54'))
                            .when(pl.col(age_col) >= 54)
                            .then(pl.lit('55-64'))
                            .when(pl.col(age_col) >= 64)
                            .then(pl.lit('65-74'))
                            .when(pl.col(age_col) >= 74)
                            .then(pl.lit('75+'))
                            .otherwise(pl.lit('75+'))
                            .alias(age_range_col)
                    )
                else:
                    df[age_range_col] = pd.cut(
                        df[age_col],
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
                return df
            return wrapper
        return decorator


@dataclass
class FindlayVoterFileConfig:
    model_config: ModelConfig = FindlayModelConfig()
    levy_election_date = datetime.strptime("2025-05-06", "%Y-%m-%d").date()
    NOVEMBER_RESULTS_COLS: ClassVar[list[str]] = ['nov_for', 'nov_against', 'nov_levy_total', 'nov_for_share', 'nov_against_share']
    PREDICTION_LEVEL_COLS: ClassVar[list[str]] = ['lean_against', 'lean_for', 'strongly_against', 'strongly_for',  'swing_against',]
    PREDICTION_TOTAL_COLS: ClassVar[list[str]] = ['total_for_share', 'total_against_share', 'total_swing_share']
    PRIMARY_COLUMNS: ClassVar[dict[str, str]] = {}
    GENERAL_COLUMNS: ClassVar[dict[str, str]] = {}
    ELECTION_DATES: ClassVar[dict[str, datetime.date]] = {}
    ELECTION_COLUMNS: ClassVar[list[str]] = []
    AGE_RANGE_SORTED: ClassVar = []
    NOVEMBER_ELECTION_NAME: ClassVar[str] = 'GENERAL-11/07/2023'

class FindlayVoterFile:
    data: pd.DataFrame
    config: ModelColumnSetup
    model_data: pd.DataFrame
    current_votes: pd.DataFrame = None
    election_results: pd.DataFrame = None

    def __init__(self):
        self.config = FindlayVoterFileConfig()
        self.load_early_votes()
        self.load_data()
        self.transform_election_data()
        self.load_election_results()
        self.create_model_dataset()

    
    @DataWrapper.format_date(FindlayEarlyVoteColumns.DATE_ENTERED, FindlayEarlyVoteColumns.DATE_RETURNED)
    def load_early_votes(self):
        vote_lists = []
        for f in FilePaths.EARLY_VOTE.glob("*.csv"):
            f = Path(f)
            df = pd.read_csv(f)
            ic(f.stem)
            if 'In Office' in f.stem:
                df[FindlayEarlyVoteColumns.VOTE_METHOD] = VotingMethod.IN_PERSON
            elif 'By Mail' in f.stem:
                df[FindlayEarlyVoteColumns.VOTE_METHOD] = VotingMethod.MAIL
            vote_lists.append(df)
        _current_votes = pd.concat(vote_lists)
        self.current_votes = _current_votes.drop_duplicates(subset=[FindlayEarlyVoteColumns.VOTER_ID])
        ic("All Early Votes: ", _current_votes[FindlayEarlyVoteColumns.VOTER_ID].count())
        ic("Unique Early Votes: ", self.current_votes[FindlayEarlyVoteColumns.VOTER_ID].nunique())
        return self.current_votes

    @DataWrapper.create_age_range(
        age_col=FindlayVoterFileColumns.AGE, 
        age_range_col=FindlayVoterFileColumns.AGE_RANGE)
    @DataWrapper.create_age_column(
        date_of_birth_col=FindlayVoterFileColumns.DATE_OF_BIRTH, 
        age_col_name=FindlayVoterFileColumns.AGE)
    @DataWrapper.format_date(
        FindlayVoterFileColumns.DATE_OF_BIRTH, 
        FindlayVoterFileColumns.REGISTRATION_DATE, 
        fmt="%Y-%m-%d")
    def load_data(self):
        _data = pl.scan_csv(list(FilePaths.DATA.glob("*.txt")), separator=',', encoding='utf8-lossy')
        _data = _data.filter(
            (pl.col(FindlayVoterFileColumns.CITY_SCHOOL_DISTRICT) == "FINDLAY CITY SD")
            & (pl.col(FindlayVoterFileColumns.COUNTY_NUMBER) == 32)
        )
        # Collect the LazyFrame to a DataFrame first
        self.data = _data.collect().to_pandas()
        return self.data
    
    def transform_election_data(self):
        self.data = self.data.fillna(pd.NA)
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
            
        # Use the variables
        _political_party = FindlayVoterFileColumns.PARTY_AFFILIATION
        self.data[_political_party] = self.data[_political_party].fillna('I')
        self.data[FindlayVoterFileColumns.WARD] = self.data[FindlayVoterFileColumns.PRECINCT_NAME].str[:-1]
        
        # Set AGE_RANGE_SORTED
        FindlayVoterFileConfig.AGE_RANGE_SORTED = [
            "18-24",
            "25-34", 
            "35-44",
            "45-54",
            "55-64",
            "65-74",
            "75+"
        ]
        
        # data['CATEGORY'] = data[_age_range].astype(str) + '-' + data[_political_party].astype(str)
        self.data['WEIGHT'] = self.data[FindlayVoterFileColumns.PARTY_AFFILIATION].map(self.config.model_config.PARTY_WEIGHTS).fillna(1.0)
        FindlayVoterFileConfig.ELECTION_DATES = {
                x: datetime.strptime(x.split("-")[1], "%m/%d/%Y").date()
                    for x in self.data.columns if "GENERAL" in x or "PRIMARY" in x
            }
        
        FindlayVoterFileConfig.PRIMARY_COLUMNS = {k: v for k, v in FindlayVoterFileConfig.ELECTION_DATES.items() if "PRIMARY" in k}
        FindlayVoterFileConfig.GENERAL_COLUMNS = {k: v for k, v in FindlayVoterFileConfig.ELECTION_DATES.items() if "GENERAL" in k}
        FindlayVoterFileConfig.ELECTION_COLUMNS = (_last4_primaries := (list(FindlayVoterFileConfig.PRIMARY_COLUMNS.keys())[-4:])) + (_last4_generals := (list(FindlayVoterFileConfig.GENERAL_COLUMNS.keys())[-4:]))
        _all_elections = list(set(FindlayVoterFileConfig.ELECTION_COLUMNS) | set(FindlayVoterFileConfig.PRIMARY_COLUMNS.keys()) | set(FindlayVoterFileConfig.GENERAL_COLUMNS.keys()))
        _all_primaries = list(FindlayVoterFileConfig.PRIMARY_COLUMNS.keys())
        _all_generals = list(FindlayVoterFileConfig.GENERAL_COLUMNS.keys())
        self.data = self.data.fillna(np.nan)
        self.data[_all_elections] = self.data[_all_elections].fillna(0).astype(bool)
        self.data[VoterScoringColumns.PRIMARY_SCORE] = self.data[_last4_primaries].sum(axis=1)
        self.data['P_SCORE_ALL'] = self.data[_all_primaries].sum(axis=1)
        self.data[VoterScoringColumns.GENERAL_SCORE] = self.data[_last4_generals].sum(axis=1)
        self.data['G_SCORE_ALL'] = self.data[_all_generals].sum(axis=1)
        self.data[FindlayVoterFileColumns.VOTED_MAY_LEVY] = self.data[FindlayVoterFileColumns.VOTER_ID].isin(self.current_votes[FindlayEarlyVoteColumns.VOTER_ID]).astype(int)
        self.data[FindlayVoterFileColumns.VOTED_NOV_LEVY] = self.data[list(FindlayVoterFileConfig.GENERAL_COLUMNS.keys())[-1]]
        self.data[FindlayVoterFileColumns.VOTED_IN_BOTH] = self.data[FindlayVoterFileColumns.VOTED_MAY_LEVY] & self.data[FindlayVoterFileColumns.VOTED_NOV_LEVY]
        return self.data
    
    def load_election_results(self):
        election_results = pd.read_csv(FilePaths.RESULTS).dropna(how='all', axis=1)
        election_results[NovemberResultsColumns.FOR] = pd.to_numeric(election_results['for'], errors='coerce').fillna(0)
        election_results[NovemberResultsColumns.AGAINST] = pd.to_numeric(election_results['against'], errors='coerce').fillna(0)
        election_results[NovemberResultsColumns.LEVY_TOTAL] = pd.to_numeric(election_results['total'], errors='coerce').fillna(0)
        election_results[FindlayEarlyVoteColumns.WARD] = election_results[FindlayEarlyVoteColumns.PRECINCT_NAME].str[:-1]
        _election_cols = list(election_results.columns)
        election_results = election_results[[_election_cols.pop(_election_cols.index(FindlayEarlyVoteColumns.WARD))] + _election_cols]
        election_results[NovemberResultsColumns.FOR_SHARE] = (election_results[NovemberResultsColumns.FOR] / election_results["total"]).round(4)
        election_results[NovemberResultsColumns.AGAINST_SHARE] = (election_results[NovemberResultsColumns.AGAINST] / election_results["total"]).round(4)
        election_results = election_results.rename(
            columns={
                "for": NovemberResultsColumns.FOR,
                "against": NovemberResultsColumns.AGAINST,
                "total": NovemberResultsColumns.LEVY_TOTAL
            }
        )
        self.election_results = election_results
    
    def create_model_dataset(self):
        self.model_data = self.data.merge(self.election_results, left_on=FindlayVoterFileColumns.PRECINCT_NAME, right_on=FindlayEarlyVoteColumns.PRECINCT_NAME)
