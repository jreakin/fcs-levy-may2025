# import marimo as mo
from prediction import (
    FindlayPredictionModel, 
    FindlayVoterFile, 
    FilePaths,
    ml_cat, 
    vote_prediction_by_age, 
    vote_prediction_by_ward, 
    vote_prediction_by_precinct, 
    plot_vote_share, 
    plot_pie_chart,
    vf_cols
)
from decision_tree_ import FindlayDecisionTree
from monte_carlo import MonteCarloVoterSimulation, MonteCarloConfig
from icecream import ic

category_data = ml_cat()
voterfile = FindlayVoterFile()
linear_model = FindlayPredictionModel(voterfile).run()
decision_tree = FindlayDecisionTree(voterfile).run()
decision_tree.run_sentiment_analysis()

# Run Monte Carlo simulation
mc_config = MonteCarloConfig(
    n_simulations=5000,  # Increase number of simulations for more accurate results
    turnout_std=0.15,    # Adjust turnout variation
    sentiment_std=0.2    # Adjust sentiment variation
)
mc_simulation = MonteCarloVoterSimulation(linear_model.data.model_data, mc_config)
mc_simulation.run_simulation()
mc_simulation.plot_distributions()
mc_simulation.print_summary_statistics()
voterfile.model_data = mc_simulation.get_voter_predictions()


# Prediction Plots

all_voters_by_age = vote_prediction_by_age(linear_model.data.model_data, linear_model.data.config)
november_voters = linear_model.data.model_data[linear_model.data.model_data[linear_model.data.config.NOVEMBER_ELECTION_NAME] == 1]
november_voters_by_age = vote_prediction_by_age(november_voters, linear_model.data.config)
november_voters_by_ward = vote_prediction_by_ward(november_voters, linear_model.data.config)
november_voters_by_precinct = vote_prediction_by_precinct(november_voters, linear_model.data.config)
november_voter_count_by_age = november_voters.groupby([vf_cols.AGE_RANGE])[vf_cols.VOTER_ID].count().reset_index().rename(columns={vf_cols.VOTER_ID: 'VOTER_COUNT'})
november_voter_count_by_age['PERCENT'] = (november_voter_count_by_age['VOTER_COUNT'] / november_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

may_voters = linear_model.data.model_data[linear_model.data.model_data['VOTED_MAY_LEVY'] == 1]
may_voters_by_age = vote_prediction_by_age(may_voters, linear_model.data.config)
may_voters_by_ward = vote_prediction_by_ward(may_voters, linear_model.data.config)
may_voters_by_precinct = vote_prediction_by_precinct(may_voters, linear_model.data.config)
may_voter_count_by_age = may_voters.groupby([vf_cols.AGE_RANGE])[vf_cols.VOTER_ID].count().reset_index().rename(columns={vf_cols.VOTER_ID: 'VOTER_COUNT'})
may_voter_count_by_age['PERCENT'] = (may_voter_count_by_age['VOTER_COUNT'] / may_voter_count_by_age['VOTER_COUNT'].sum() * 100).round(2)

ic(november_voters_by_ward.sum())

plot_vote_share(november_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_TOTAL_COLS, 'November Voters by Age')
plot_vote_share(november_voters_by_ward, 'WARD', linear_model.config.PREDICTION_TOTAL_COLS, 'November Voters by Ward')
plot_vote_share(may_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_TOTAL_COLS, 'May Voters by Age')
plot_vote_share(may_voters_by_ward, 'WARD', linear_model.config.PREDICTION_TOTAL_COLS, 'May Voters by Ward')

plot_vote_share(november_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_LEVEL_COLS, 'November Voters by Age')
plot_vote_share(november_voters_by_ward, 'WARD', linear_model.config.PREDICTION_LEVEL_COLS, 'November Voters by Ward')
plot_vote_share(may_voters_by_age, 'AGE_RANGE', linear_model.config.PREDICTION_LEVEL_COLS, 'May Voters by Age')
plot_vote_share(may_voters_by_ward, 'WARD', linear_model.config.PREDICTION_LEVEL_COLS, 'May Voters by Ward')


nov_by_level = november_voters.groupby('vote_prediction')[vf_cols.VOTER_ID].count()
nov_by_generic = november_voters.groupby('generic_vote_prediction')[vf_cols.VOTER_ID].count()
nov_by_semi_generic = november_voters.groupby('semi_generic_prediction')[vf_cols.VOTER_ID].count()

may_by_level = may_voters.groupby('vote_prediction')[vf_cols.VOTER_ID].count()
may_by_generic = may_voters.groupby('generic_vote_prediction')[vf_cols.VOTER_ID].count()
may_by_semi_generic = may_voters.groupby('semi_generic_prediction')[vf_cols.VOTER_ID].count()

plot_pie_chart(nov_by_level, 'November Voters by Vote Prediction')
plot_pie_chart(nov_by_semi_generic, 'November Voters by Semi-Generic Vote Prediction')
plot_pie_chart(nov_by_generic, 'November Voters by Generic Vote Prediction')

plot_pie_chart(may_by_level, 'May Voters by Vote Prediction')
plot_pie_chart(may_by_semi_generic, 'May Voters by Semi-Generic Vote Prediction')
plot_pie_chart(may_by_generic, 'May Voters by Generic Vote Prediction')


# Exports

november_voters_by_ward.to_csv(FilePaths.PREDICTION_FOLDER / 'november_voters_by_ward.csv', index=False)
may_voters_by_ward.to_csv(FilePaths.PREDICTION_FOLDER / 'may_voters_by_ward.csv', index=False)
november_voters_by_age.to_csv(FilePaths.PREDICTION_FOLDER / 'november_voters_by_age.csv', index=False)
may_voters_by_age.to_csv(FilePaths.PREDICTION_FOLDER / 'may_voters_by_age.csv', index=False)
november_voters_by_precinct.to_csv(FilePaths.PREDICTION_FOLDER / 'november_voters_by_precinct.csv', index=False)
may_voters_by_precinct.to_csv(FilePaths.PREDICTION_FOLDER / 'may_voters_by_precinct.csv', index=False)
