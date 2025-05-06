from enum import StrEnum
from typing import Protocol
from datetime import datetime
import pandas as pd
import abc


# ===== Voter File Protcols =====

class VoterScoringColumns:
    PRIMARY_SCORE = 'P_SCORE'
    GENERAL_SCORE = 'G_SCORE'

class VotingMethod(StrEnum):
    IN_PERSON = 'In-Person'
    MAIL = 'Mail'

# ===== Model Protocols =====


class ModelConfig(Protocol):
    PARTY_WEIGHTS: dict[str, float]


class PredictionGranularTiers(Protocol):
    STRONGLY_AGAINST: str
    LEAN_AGAINST: str
    SWING_AGAINST: str
    LEAN_FOR: str
    STRONGLY_FOR: str


class PredictionTotalTiers(Protocol):
    TOTAL_FOR_SHARE: str
    TOTAL_AGAINST_SHARE: str
    TOTAL_SWING_SHARE: str

class ModelColumnSetup(Protocol):
    NOVEMBER_RESULTS_COLS: list[str]
    PREDICTION_LEVEL_COLS: list[str]
    PREDICTION_TOTAL_COLS: list[str]
    PRIMARY_COLUMNS: dict[str, str]
    GENERAL_COLUMNS: dict[str, str]
    ELECTION_DATES: dict[str, datetime.date]
    ELECTION_COLUMNS: list[str]
    AGE_RANGE_SORTED: list[str]
    NOVEMBER_ELECTION_NAME: str = None


class LinearModelFeatureLists(Protocol):
    category_features: list[str]
    high_cardinality_features: list[str]
    numerical_features: list[str]
    interaction_features: list[str]
    all_features: list[str]

class VoterFileData(abc.ABC):
    data: pd.DataFrame
    config: ModelColumnSetup
    model_data: pd.DataFrame
    current_votes: pd.DataFrame
    election_results: pd.DataFrame

    @property
    def model_config(self) -> ModelConfig:
        return self.config.MODEL_CONFIG
    

    @property
    def election_dates(self) -> dict[str, datetime.date]:
        return self.config.ELECTION_DATES
    
    @election_dates.setter
    def election_dates(self, value: dict[str, datetime.date]):
        self.config.ELECTION_DATES.update(value)

    @property
    def primary_columns(self) -> dict[str, str]:
        return self.config.PRIMARY_COLUMNS
    
    @primary_columns.setter
    def primary_columns(self, value: dict[str, str]):
        self.config.PRIMARY_COLUMNS.update(value)

    @property
    def general_columns(self) -> dict[str, str]:
        return self.config.GENERAL_COLUMNS
    
    @general_columns.setter
    def general_columns(self, value: dict[str, str]):
        self.config.GENERAL_COLUMNS.update(value)
    
    @property
    def age_range_sorted(self) -> list[str]:
        return self.config.AGE_RANGE_SORTED
    
    @age_range_sorted.setter
    def age_range_sorted(self, value: list[str]):
        self.config.AGE_RANGE_SORTED = value
    
    @property
    def election_columns(self) -> list[str]:
        return self.config.ELECTION_COLUMNS
    
    @election_columns.setter
    def election_columns(self, value: list[str]):
        self.config.ELECTION_COLUMNS = value
    
    @property
    def weighted_precinct_category_counts(self) -> pd.DataFrame:
        return self.config.WEIGHTED_PRECINCT_CATEGORY_COUNTS


class ModelDataStartingPoint(Protocol):
    model_data: pd.DataFrame
    config: ModelColumnSetup