import pandas as pd
import matplotlib.pyplot as plt

from data_loader import FindlayVoterFile, FindlayVoterFileConfig as vf_config, NovemberResultsColumns as nov_results, FilePaths
from config import FindlayLinearModelFeatureLists as ml_cat


print("Loading data and initializing November modeling...")


category_data = ml_cat()
voterfile = FindlayVoterFile()

election_results = voterfile.election_results
election_results_against_mean = election_results['nov_against_share'].mean()
election_results_for_mean = election_results['nov_for_share'].mean()
election_results_against_std = election_results['nov_against_share'].std()
thresholds = election_results['nov_for_share'].quantile([0.2, 0.4, 0.6, 0.8])

election_results['nov_for_share'].hist()
plt.show()

m_data = voterfile.model_data
november = m_data[m_data['VOTED_NOV_LEVY'] == 1]
may = m_data[m_data['VOTED_MAY_LEVY'] == 1]
ward_turnout_expectations = m_data.groupby(['WARD', 'PRECINCT_NAME']).agg(
    {
        'primary_precinct_turnout_mean': 'mean',
        'primary_ward_turnout_mean': 'mean',
        'general_precinct_turnout_mean': 'mean',
        'general_ward_turnout_mean': 'mean',
     }
)
ward_turnout_expectations["eligible_voters"] = m_data.groupby(["ward", "precinct"]).size().values

def round_and_cast(_col: pd.Series):
    return _col.round().astype(int)

def expected_turnout_func(x):
    return round_and_cast(x * ward_turnout_expectations["eligible_voters"])

ward_turnout_expectations['primary_precinct_turnout_count'] = expected_turnout_func(ward_turnout_expectations['primary_precinct_turnout_mean'])
ward_turnout_expectations['primary_ward_turnout_count'] = expected_turnout_func(ward_turnout_expectations['primary_ward_turnout_mean'])
ward_turnout_expectations['general_precinct_turnout_count'] = expected_turnout_func(ward_turnout_expectations['general_precinct_turnout_mean'])
ward_turnout_expectations['general_ward_turnout_count'] = expected_turnout_func(ward_turnout_expectations['general_ward_turnout_mean'])


# Category Features
m_data[age_range_cat := 'AGE_RANGE_CAT'] = pd.Categorical(m_data['AGE_RANGE'], categories=sorted(m_data['AGE_RANGE'].unique()), ordered=True)
m_data[party_cat := 'PARTY_CAT'] = pd.Categorical(m_data['PARTY_AFFILIATION'], categories=['D', 'I', 'R'], ordered=True)
ml_cat.category_features.extend([age_range_cat, party_cat])


# Interaction Features
m_data[age_ward := 'AGE_WARD'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['WARD'].astype(str)
m_data[age_precinct := 'AGE_PRECINCT'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['PRECINCT_NAME'].astype(str)
m_data[age_party := 'AGE_PARTY'] = m_data['AGE_RANGE'].astype(str) + '-' + m_data['PARTY_AFFILIATION'].astype(str)
m_data[p_score_last4_cat := 'P_SCORE_LAST4_CAT'] = pd.cut(
    m_data['P_SCORE'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[g_score_last4_cat := 'G_SCORE_LAST4_CAT'] = pd.cut(
    m_data['G_SCORE'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[p_score_all_cat := 'P_SCORE_ALL_CAT'] = pd.cut(
    m_data['P_SCORE_ALL'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[g_score_all_cat := 'G_SCORE_ALL_CAT'] = pd.cut(
    m_data['G_SCORE_ALL'],
    bins=5,
    labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
).astype(int)
m_data[p_score_last4_ward := 'P_SCORE_LAST4_WARD'] = m_data[p_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[g_score_last4_ward := 'G_SCORE_LAST4_WARD'] = m_data[g_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[p_score_all_ward := 'P_SCORE_ALL_WARD'] = m_data[p_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[g_score_all_ward := 'G_SCORE_ALL_WARD'] = m_data[g_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str)
m_data[p_score_last4_age_ward_precinct := 'P_SCORE_LAST4_AGE_WARD_PRECINCT'] = m_data[p_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[g_score_last4_age_ward_precinct := 'G_SCORE_LAST4_AGE_WARD_PRECINCT'] = m_data[g_score_last4_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[p_score_all_age_ward_precinct := 'P_SCORE_ALL_AGE_WARD_PRECINCT'] = m_data[p_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
m_data[g_score_all_age_ward_precinct := 'G_SCORE_ALL_AGE_WARD_PRECINCT'] = m_data[g_score_all_cat].astype(str) + '-' + m_data[age_ward].astype(str) + '-' + m_data[age_precinct].astype(str)
ml_cat.interaction_features.extend([
    age_ward,
    age_precinct,
    age_party,
    p_score_last4_cat,
    g_score_last4_cat,
    p_score_all_cat,
    g_score_all_cat,
    p_score_last4_ward,
    g_score_last4_ward,
    p_score_all_ward,
    g_score_all_ward,
    p_score_last4_age_ward_precinct,
    g_score_last4_age_ward_precinct,
    p_score_all_age_ward_precinct,
    g_score_all_age_ward_precinct,
])

nov_age = (
november['AGE_RANGE']
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
    .rename(columns={
        'proportion': 'nov_count'
    }))

may_age = (
    may['AGE_RANGE']
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
    .rename(columns={
        'proportion': 'may_count'
    }))

merge_ages = nov_age.merge(may_age, on='AGE_RANGE')

merge_by_ward = (
    november['WARD']
    .value_counts(normalize=True)
    .to_frame()
    .merge(
        may['WARD']
        .value_counts(normalize=True)
        .to_frame(),
        on='WARD')
    .rename(
        columns={
            'proportion_x': 'nov_voters',
            'proportion_y': 'may_voters'
        }
    )
).sort_index()

nov_ward_results = (
    voterfile.election_results
    .groupby('ward')
    .agg(
        {
            'nov_ward_for_count': 'sum',
            'nov_ward_against_count': 'sum',
            'nov_levy_total': 'sum',
            'nov_ward_for_share': 'mean',
            'nov_ward_against_share': 'mean',
            'nov_ward_turnout': 'mean',

        }
    )
    .reset_index()
)

may_election_results = (
    m_data
    .groupby(['WARD', 'PRECINCT_NAME', 'AGE_RANGE'])
    .agg(
        {
            'VOTED_NOV_LEVY': 'sum',
            'VOTED_MAY_LEVY': 'sum',
            'SOS_VOTERID': 'count'
        })
    .reset_index()
    .rename(
        columns={
            'VOTED_MAY_LEVY': 'may_total_voted',
            'VOTED_NOV_LEVY': 'nov_total_voted',
            'SOS_VOTERID': 'total_registered_voters'
        }
    )
)

for precinct in m_data['PRECINCT_NAME'].unique():
    _for_share = m_data[
    m_data['PRECINCT_NAME'] == precinct]['nov_for_share'].mean()

    _against_share = m_data[
    m_data['PRECINCT_NAME'] == precinct
    ]['nov_against_share'].mean()

    for age_range in m_data['AGE_RANGE'].unique():
        _for_share = m_data[
        m_data['PRECINCT_NAME'] == precinct][
        m_data['AGE_RANGE'] == age_range
        ]['nov_for_share'].mean()

        _against_share = m_data[
        m_data['PRECINCT_NAME'] == precinct][
        m_data['AGE_RANGE'] == age_range
        ]['nov_against_share'].mean()
        may_election_results.loc[may_election_results['PRECINCT_NAME'] == precinct, 'may_precinct_for_share'] = (
    may_election_results[
            may_election_results['PRECINCT_NAME'] == precinct
            ]['may_total_voted'] * _for_share).round()
        may_election_results.loc[may_election_results['PRECINCT_NAME'] == precinct, 'may_precinct_against_share'] = (
    may_election_results[
            may_election_results['PRECINCT_NAME'] == precinct
            ]['may_total_voted'] * _against_share).round()

may_ward_results = (
    may_election_results
    .groupby('WARD')
    .agg(
        {
            'may_total_voted': 'sum',
            'total_registered_voters': 'sum'
        }
    )
    .reset_index()
)

merged_results = (
    nov_ward_results
    .merge(
        may_ward_results,
        right_on='WARD',
        left_on='ward'
    )
)

merged_results['may_ward_turnout'] = (
    merged_results['may_total_voted'] /
    merged_results['total_registered_voters']).round(4)

merged_results['may_votes_FOR'] = (
    merged_results['may_total_voted'] *
    merged_results['nov_ward_for_share']).astype(int)

merged_results['may_votes_pct_FOR'] = (
    merged_results['may_votes_FOR'] /
    merged_results['may_total_voted']).round(4)

merged_results['may_votes_AGAINST'] = (
    merged_results['may_total_voted'] *
    merged_results['nov_ward_against_share']).astype(int)

ward_turnout_count = ward_turnout_expectations.groupby('WARD').agg(
    {
        'primary_ward_turnout_count': 'sum',
        'primary_ward_turnout_mean': 'first',
    }
)
ward_turnout_count['primary_ward_turnout_mean'] = ward_turnout_count['primary_ward_turnout_mean'].round(4)
merged_results = merged_results.merge(ward_turnout_count, right_on='WARD', left_on='ward')
merged_results['may_est_total_for'] = round_and_cast(merged_results['primary_ward_turnout_count'] * merged_results['nov_ward_for_share'])
merged_results['may_est_total_against'] = round_and_cast(merged_results['primary_ward_turnout_count'] * merged_results['nov_ward_against_share'])
merged_results['may_est_percent_for'] = (merged_results['may_est_total_for'] / merged_results['primary_ward_turnout_count']).round(4)
merged_results['may_est_percent_against'] = (merged_results['may_est_total_against'] / merged_results['primary_ward_turnout_count']).round(4)
merged_results['better_than_nov'] = merged_results['nov_ward_for_share'] < merged_results['may_est_percent_for']
merged_results['winning_ward'] = merged_results['may_est_percent_for'] >= .5