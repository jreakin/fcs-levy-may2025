from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path


class FilePaths:
    DOWNLOAD_PATH: Path = Path.home() / 'Downloads'
    VOTERFILE_PATH: Path = Path.home() / 'PyCharmProjects' / 'state-voterfiles'
    DATA_PATH: Path = Path(__file__).parent / 'data'
    PREDICTION_FOLDER: Path = DATA_PATH / 'may25_predictions'
    IMAGE_PATH: Path = DATA_PATH / 'images'

    RESULTS: Path = DATA_PATH / 'NOV24-FCS-TAX.csv'
    EARLY_VOTE: Path = DATA_PATH / 'may26_ev'
    DATA: Path = VOTERFILE_PATH / "data/ohio/voterfile/ohio-statewide"

class FindlayVoterFileColumns:
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
    VOTER_ID = 'SOS_VOTERID'
    VOTED_MAY_LEVY = 'VOTED_MAY_LEVY'
    VOTED_NOV_LEVY = 'VOTED_NOV_LEVY'
    VOTED_IN_BOTH = 'VOTED_IN_BOTH'

class FindlayEarlyVoteColumns:
    VOTER_ID = 'STATE ID#'
    DATE_ENTERED = 'DATE ENTERED:'
    DATE_RETURNED = 'DATE RETURNED:'
    PRECINCT_NAME = 'precinct'
    WARD = 'ward'
    VOTE_METHOD = 'Vote Method'



class FindlayModelConfig:
    levy_election_date = datetime.strptime("2025-05-06", "%Y-%m-%d").date()
    PARTY_WEIGHTS = {
        'D': 1.5,  # Democrats weighted highest
        'R': 1.0,  # Republicans weighted higher than Independents
        'I': 1.2   # Independents as baseline
    }

class FindlayPredictionGranularTiers:
    STRONGLY_AGAINST = 'strongly_against'
    LEAN_AGAINST = 'lean_against'
    SWING_AGAINST = 'swing_against'
    SWING = 'swing'
    SWING_FOR = 'swing_for'
    LEAN_FOR = 'lean_for'
    STRONGLY_FOR = 'strongly_for'


class FindlayPredictionTotalTiers:
    TOTAL_FOR_SHARE = 'total_for_share'
    TOTAL_AGAINST_SHARE = 'total_against_share'
    TOTAL_SWING_SHARE = 'total_swing_share'

class NovemberResultsColumns:
    FOR = 'nov_for'
    AGAINST = 'nov_against'
    LEVY_TOTAL = 'nov_levy_total'
    FOR_SHARE = 'nov_for_share'
    AGAINST_SHARE = 'nov_against_share'

class FindlayVoterFileConfig:
    NOVEMBER_RESULTS_COLS = ['nov_for', 'nov_against', 'nov_levy_total', 'nov_for_share', 'nov_against_share']
    PREDICTION_LEVEL_COLS = [
        'strongly_against',
        'lean_against',
        'swing_against',
        'swing',
        'swing_for',
        'lean_for',
        'strongly_for'
    ]
    PREDICTION_TOTAL_COLS = [
        'total_for_share',
        'total_against_share',
        'total_swing_share'
    ]
    PRIMARY_COLUMNS = {}
    GENERAL_COLUMNS = {}
    ELECTION_DATES = {}
    ELECTION_COLUMNS = []   
    AGE_RANGE_SORTED = []
    NOVEMBER_ELECTION_NAME = None

class FindlayLinearModelFeatureLists:
    category_features = []
    high_cardinality_features = []
    numerical_features = []
    interaction_features = []
    all_features = []
