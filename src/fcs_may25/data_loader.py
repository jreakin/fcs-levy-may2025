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
from typing import ClassVar
import warnings
from contextlib import ExitStack

import polars as pl
from datetime import datetime
import numpy as np
import pandas as pd
from icecream import ic

warnings.filterwarnings('ignore')


class DataWrapper:

    @staticmethod
    def format_date(*date_columns):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                df = func(self, *args, **kwargs)
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
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
        self.current_votes = pd.concat(vote_lists).drop_duplicates(subset=[FindlayEarlyVoteColumns.VOTER_ID])
        return self.current_votes

    def load_data(self):
        _data = pl.scan_csv(list(FilePaths.DATA.glob("*.txt")), separator=',', encoding='utf8-lossy')
        _data = _data.filter(
            (pl.col(FindlayVoterFileColumns.CITY_SCHOOL_DISTRICT) == "FINDLAY CITY SD")
            & (pl.col(FindlayVoterFileColumns.COUNTY_NUMBER) == 32)
        )
        _data = _data.with_columns(
            pl.col(FindlayVoterFileColumns.DATE_OF_BIRTH).str.strptime(pl.Datetime, format="%Y-%m-%d"),
            pl.col(FindlayVoterFileColumns.REGISTRATION_DATE).str.strptime(pl.Datetime, format="%Y-%m-%d"),
        )
        
        with ExitStack():
            # Define all variables within the context
            _political_party = FindlayVoterFileColumns.PARTY_AFFILIATION
            _age = FindlayVoterFileColumns.AGE
            _age_range = FindlayVoterFileColumns.AGE_RANGE

            _data = _data.with_columns(
                # Calculate age in years
            (
                (pl.lit(datetime.now()) - pl.col(FindlayVoterFileColumns.DATE_OF_BIRTH))
                    .dt.total_days()
                    // 365  # Integer division to get approximate years
                )
                .cast(pl.Int64)
                .alias(_age)
                    )
            data = _data.collect().to_pandas().fillna(pd.NA)
            data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
            
            # Use the variables
            data[_political_party] = data[_political_party].fillna('I')
            data[_age] = data[_age].fillna(data[_age].mean())
            data[_age_range] = pd.cut(
                data[_age],
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
            FindlayVoterFileConfig.AGE_RANGE_SORTED = sorted(data[_age_range].unique().tolist())
            data[FindlayVoterFileColumns.WARD] = data[FindlayVoterFileColumns.PRECINCT_NAME].str[:-1]
            data['CATEGORY'] = data[_age_range].astype(str) + '-' + data[_political_party].astype(str)
            data['WEIGHT'] = data[_political_party].map(self.config.model_config.PARTY_WEIGHTS).fillna(1.0)
        FindlayVoterFileConfig.ELECTION_DATES = {
                x: datetime.strptime(x.split("-")[1], "%m/%d/%Y").date()
                    for x in data.columns if "GENERAL" in x or "PRIMARY" in x
            }
        
        FindlayVoterFileConfig.PRIMARY_COLUMNS = {k: v for k, v in FindlayVoterFileConfig.ELECTION_DATES.items() if "PRIMARY" in k}
        FindlayVoterFileConfig.GENERAL_COLUMNS = {k: v for k, v in FindlayVoterFileConfig.ELECTION_DATES.items() if "GENERAL" in k}
        FindlayVoterFileConfig.ELECTION_COLUMNS = (_last4_primaries := (list(FindlayVoterFileConfig.PRIMARY_COLUMNS.keys())[-4:])) + (_last4_generals := (list(FindlayVoterFileConfig.GENERAL_COLUMNS.keys())[-4:]))
        data[FindlayVoterFileConfig.ELECTION_COLUMNS] = data[FindlayVoterFileConfig.ELECTION_COLUMNS].fillna(0).astype(bool)
        data = data.fillna(np.nan)
        data[VoterScoringColumns.PRIMARY_SCORE] = data[_last4_primaries].sum(axis=1).round(2)
        data[VoterScoringColumns.GENERAL_SCORE] = data[_last4_generals].sum(axis=1).round(2)
        self.data = data
        self.data[FindlayVoterFileColumns.VOTED_MAY_LEVY] = self.data[FindlayVoterFileColumns.VOTER_ID].isin(self.current_votes[FindlayEarlyVoteColumns.VOTER_ID]).astype(int)
        self.data[FindlayVoterFileColumns.VOTED_IN_BOTH] = self.data[FindlayVoterFileColumns.VOTED_MAY_LEVY] & self.data[list(FindlayVoterFileConfig.GENERAL_COLUMNS.keys())[-1]]
        return self
    
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
