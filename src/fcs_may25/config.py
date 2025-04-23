from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path

class FilePaths(Enum):
    DOWNLOAD_PATH = Path.home() / 'Downloads'
    VOTERFILE_PATH = Path.home() / 'PyCharmProjects' / 'state-voterfiles'
    DATA_PATH = Path(__file__).parent / 'data'
    PREDICTION_FOLDER = DATA_PATH / 'may25_predictions'
    IMAGE_PATH = DATA_PATH / 'images'

    RESULTS = DATA_PATH / 'NOV24-FCS-TAX.csv'
    EARLY_VOTE = DATA_PATH / 'may26_ev'
    DATA = VOTERFILE_PATH / "data/ohio/voterfile/ohio-statewide"

class FindlayVoterFileColumns(StrEnum):
    PARTY_AFFILIATION = 'PARTY_AFFILIATION'
    AGE = 'AGE'
    AGE_RANGE = 'AGE_RANGE'
    PRECINCT_NAME = 'PRECINCT_NAME'
    WARD = 'WARD'
    PRECINCT = 'PRECINCT'
    REGISTRATION_DATE = 'REGISTRATION_DATE'
    DATE_OF_BIRTH = 'DATE_OF_BIRTH'
    CITY_SCHOOL_DISTRICT = 'CITY_SCHOOL_DISTRICT'
    COUNTY_NUMBER = 'COUNTY_NUMBER'
    COUNTY_NAME = 'COUNTY_NAME'
    STATE_ID = 'STATE_ID'

class FindlayMLModelCategories:
    AGE_RANGE_CAT = 'AGE_RANGE_CAT'
    PARTY_CAT = 'PARTY_CAT'
    AGE_WARD = 'AGE_WARD'
    AGE_PRECINCT = 'AGE_PRECINCT'
    AGE_PARTY = 'AGE_PARTY'
    P_SCORE = 'P_SCORE'
    G_SCORE = 'G_SCORE'
    AGE = 'AGE'
    _interaction_features = []
    _category_features = []
    _high_cardinality_features = []
    _numerical_features = []

    @property
    def interaction_features(self):
        self._interaction_features = [self.AGE_WARD, self.AGE_PRECINCT, self.AGE_PARTY]
        return self._interaction_features
    
    @interaction_features.setter
    def interaction_features(self, value):
        self._interaction_features.extend(value)
    
    @property
    def category_features(self):
        self._category_features = [self.PARTY_CAT, self.AGE_RANGE_CAT]
        return self._category_features
    
    @property
    def high_cardinality_features(self):
        self._high_cardinality_features = [self.PRECINCT_NAME, self.WARD]
        return self._high_cardinality_features
    
    @property
    def numerical_features(self):
        self._numerical_features = [self.P_SCORE, self.G_SCORE, self.AGE]
        return self._numerical_features
    
    @numerical_features.setter
    def numerical_features(self, value):
        self._numerical_features.extend(value)

class FindlayModelConfig:
    levy_election_date = datetime.strptime("2025-05-06", "%Y-%m-%d").date()
    PARTY_WEIGHTS = {
        'D': 1.5,  # Democrats weighted highest
        'R': 1.0,  # Republicans weighted higher than Independents
        'I': 1.2   # Independents as baseline
    }

class FindlayVoterFileConfig:
    MODEL_CONFIG = FindlayModelConfig()
    NOVEMBER_RESULTS_COLS = ['nov_for', 'nov_against', 'nov_levy_total', 'nov_for_share', 'nov_against_share']
    PREDICTION_LEVEL_COLS = ['lean_against', 'lean_for', 'strongly_against', 'strongly_for',  'swing_against',]
    PREDICTION_TOTAL_COLS = ['total_for_share', 'total_against_share', 'total_swing_share']
    PRIMARY_COLUMNS = {}
    GENERAL_COLUMNS = {}
    ELECTION_DATES = {}
    ELECTION_COLUMNS = []
    AGE_RANGE_SORTED = []
    NOVEMBER_ELECTION_NAME = None

    @property
    def model_config(self) -> FindlayModelConfig:
        return self.MODEL_CONFIG
    