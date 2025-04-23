from typing import Protocol
from datetime import datetime
import pandas as pd

class ModelConfig(Protocol):
    levy_election_date: datetime.date
    PARTY_WEIGHTS: dict[str, float]

class VoterFileConfig(Protocol):
    MODEL_CONFIG: ModelConfig
    NOVEMBER_RESULTS_COLS: list[str]
    PREDICTION_LEVEL_COLS: list[str]
    PREDICTION_TOTAL_COLS: list[str]
    PRIMARY_COLUMNS: dict[str, str]
    GENERAL_COLUMNS: dict[str, str]
    ELECTION_DATES: dict[str, datetime.date]
    ELECTION_COLUMNS: list[str]
    AGE_RANGE_SORTED: list[str]
    NOVEMBER_ELECTION_NAME: str = None

class VoterFileData(Protocol):
    data: pd.DataFrame
    config: VoterFileConfig
    model_data: pd.DataFrame
    weighted_data: pd.DataFrame
    primary_scores: pd.DataFrame
    weighted_precinct_category_counts: pd.DataFrame
    turnout_data: pd.DataFrame
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
        self.config.ELECTION_DATES = value

    @property
    def primary_columns(self) -> dict[str, str]:
        return self.config.PRIMARY_COLUMNS
    
    @primary_columns.setter
    def primary_columns(self, value: dict[str, str]):
        self.config.PRIMARY_COLUMNS = value

    @property
    def general_columns(self) -> dict[str, str]:
        return self.config.GENERAL_COLUMNS
    
    @general_columns.setter
    def general_columns(self, value: dict[str, str]):
        self.config.GENERAL_COLUMNS = value
    
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
