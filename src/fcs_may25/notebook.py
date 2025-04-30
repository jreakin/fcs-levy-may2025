import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from icecream import ic
    import numpy as np
    from category_encoders import TargetEncoder
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
    from sklearn.compose import ColumnTransformer
    import seaborn as sns
    import pandas as pd
    from functools import partial
    import matplotlib.pyplot as plt
    from typing import Optional

    from data_loader import FindlayVoterFile, FindlayVoterFileConfig as vf_config, NovemberResultsColumns as nov_results
    from config import FindlayLinearModelFeatureLists as ml_cat
    from prediction import (
        FindlayPredictionModel,
        test_max_iter_impact,
        plot_regression_results,
        plot_feature_importance
    )
    return (
        ColumnTransformer,
        FindlayPredictionModel,
        FindlayVoterFile,
        LinearRegression,
        OneHotEncoder,
        Optional,
        StandardScaler,
        TargetEncoder,
        ic,
        mean_squared_error,
        ml_cat,
        mo,
        nov_results,
        np,
        partial,
        pd,
        plot_feature_importance,
        plot_regression_results,
        plt,
        r2_score,
        sns,
        test_max_iter_impact,
        tqdm,
        train_test_split,
        vf_config,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Findlay City Schools Tax Levy: Turnout Report
        Prepared by: John R. Eakin, Abstract Data
        """
    )
    return


@app.cell
def _(FindlayVoterFile):
    voterfile = FindlayVoterFile()
    m_data = voterfile.model_data
    return m_data, voterfile


@app.cell
def _(Optional, partial, pd, plt):
    # Function Setup

    def create_pie_chart(
        df: pd.DataFrame | pd.Series, 
        title: str, 
        ax: Optional[plt.Axes] = None,
        labels_map: dict = None  # Add new parameter for label mapping
    ):
        data = df.value_counts(normalize=True).sort_index()

        # If labels_map is provided, rename the index
        if labels_map:
            data.index = data.index.map(labels_map)

        return data.plot(
            kind='pie',
            autopct='%1.1f%%',
            ax=ax,
            title=title
        )

    new_subplot = partial(plt.subplots, 1, 2, figsize=(15,7))
    return create_pie_chart, new_subplot


@app.cell
def _(mo):
    mo.md(r"""# Turnout Comparisons""")
    return


@app.cell
def _(mo):
    mo.md(r"""## City-Wide Age Ranges for Registered Voters""")
    return


@app.cell
def _(create_pie_chart, m_data, new_subplot, plt):
    fig0, (ax01, ax02) = new_subplot()
    create_pie_chart(m_data['AGE_RANGE'], title='All Voters', ax=ax01)
    m_data['AGE_RANGE'].value_counts().sort_index().plot(
        kind='bar', 
        ylabel='Count',
        xlabel='Age Range',
        title='City-Wide Voter Count By Age Range',
        ax=ax02)
    plt.gca()
    return ax01, ax02, fig0


@app.cell
def _(m_data):
    november = m_data[m_data['VOTED_NOV_LEVY'] == True]
    may = m_data[m_data['VOTED_MAY_LEVY'] == True]
    return may, november


@app.cell
def _(mo):
    mo.md(r"""## November vs May: Turnout By Age Range""")
    return


@app.cell
def _(create_pie_chart, may, new_subplot, november, plt):
    fig1, (ax1, ax2) = new_subplot()

    create_pie_chart(november['AGE_RANGE'], title='November', ax=ax1)
    create_pie_chart(may['AGE_RANGE'], title='May', ax=ax2)
    plt.gca()
    return ax1, ax2, fig1


@app.cell
def _(may, november, plt):
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
    merge_ages.plot(
        kind='bar', 
        y=['nov_count', 'may_count'],
        x='AGE_RANGE',
        xlabel='Age Range',
        ylabel='Percent',
        title='Turnout Percentage By Age Range'
    )
    plt.gca()
    return may_age, merge_ages, nov_age


@app.cell
def _(mo):
    mo.md(r"""## Turnout By Ward""")
    return


@app.cell
def _(create_pie_chart, may, new_subplot, november, plt):
    fig2, (ax3, ax4) = new_subplot()

    create_pie_chart(november['WARD'], title='November', ax=ax3)
    create_pie_chart(may['WARD'], title='May', ax=ax4)
    plt.gca()
    return ax3, ax4, fig2


@app.cell
def _(voterfile):
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
    return (nov_ward_results,)


@app.cell
def _(m_data, nov_ward_results):
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
        merged_results['may_total_voted']
    ).round(4)
    return (
        age_range,
        may_election_results,
        may_ward_results,
        merged_results,
        precinct,
    )


@app.cell
def _(mo):
    mo.md(r"""## Prediction Modeling""")
    return


@app.cell
def _(november):
    november_set = november
    y_pseudo = november_set['nov_for'] / november_set['total']
    return november_set, y_pseudo


@app.cell
def _(ColumnTransformer, OneHotEncoder, StandardScaler, TargetEncoder, ml_cat):
    # # Convert categorical columns to dummy variables
    # categorical_dummies = pd.get_dummies(
    #     november_set[ml_cat.category_features + ml_cat.interaction_features],
    #     drop_first=True, # Drop first category to avoid multicollinearity
    #     dtype=float)      # Convert to float for the model


    preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), ml_cat.category_features),
                    ("high_card", TargetEncoder(), ml_cat.high_cardinality_features),
                    ("num", StandardScaler(), ml_cat.numerical_features),
                    ("interaction", TargetEncoder(), ml_cat.interaction_features),
                ]
            )
    return (preprocessor,)


@app.cell
def _(
    LinearRegression,
    november_set,
    np,
    pd,
    preprocessor,
    train_test_split,
    y_pseudo,
):
    # Combine all features
    X = preprocessor.fit_transform(november_set, november_set['nov_for_share'])
    feature_names = preprocessor.get_feature_names_out()

    y = y_pseudo  # Using the same target variable as before

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred_test = model.predict(X_test)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'against': np.abs(model.coef_[0]),
        'swing': np.abs(model.coef_[1]),
        'for': np.abs(model.coef_[2])
    })
    feature_importance['overall_importance'] = feature_importance[['against', 'swing', 'for']].mean(axis=1)
    feature_importance = feature_importance.sort_values('overall_importance', ascending=False)
    return (
        X,
        X_test,
        X_train,
        feature_importance,
        feature_names,
        model,
        y,
        y_pred_test,
        y_test,
        y_train,
    )


@app.cell
def _(plot_regression_results, y_pred_test, y_test):
    # Now plot the results
    plot_regression_results(y_test, y_pred_test, 'Linear Regression')
    return


@app.cell
def _(feature_names, model, plot_feature_importance):
    plot_feature_importance(model, feature_names)
    return


@app.cell
def _(X, model, november_set, y_pseudo):
    all_predictions = model.predict(X)

    november_set['prediction_for_share'] = all_predictions
    november_set['prediction_residual'] = y_pseudo - all_predictions
    return (all_predictions,)


@app.cell
def _(mo):
    mo.md(r"""### Prediction Density""")
    return


@app.cell
def _(november_set, plt, sns):
    # Create KDE plot for prediction distributions
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=november_set['prediction_for_share'], 
        label='Prediction Share', 
        alpha=0.7
    )
    return


@app.cell
def _(plt):
    actual_result = 0.4640  # November election result
    std_dev = 0.07  # Standard deviation from November results
    lean_margin = std_dev  # 1 standard deviation for lean (±7%)
    strong_margin = std_dev * 2  # 2 standard deviations for strong (±14%)

    # Add vertical line for actual result
    plt.axvline(x=actual_result, color='red', linestyle='-', alpha=0.8, label='Actual November Result (46.40%)')

    # Add threshold lines based on margins from actual result
    plt.axvline(x=actual_result - strong_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=actual_result - lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=actual_result + lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=actual_result + strong_margin, color='gray', linestyle='--', alpha=0.5)

    # Add labels
    plt.text(actual_result - strong_margin, plt.ylim()[1], f'\nStrongly Against\n(<{(actual_result-strong_margin)*100:.1f}%)', rotation=90, ha='right')
    plt.text(actual_result - lean_margin, plt.ylim()[1], f'\nLean Against\n(<{(actual_result-lean_margin)*100:.1f}%)', rotation=90, ha='right')
    plt.text(actual_result + lean_margin, plt.ylim()[1], f'\nLean For\n(>{(actual_result+lean_margin)*100:.1f}%)', rotation=90, ha='right')
    plt.text(actual_result + strong_margin, plt.ylim()[1], f'\nStrongly For\n(>{(actual_result+strong_margin)*100:.1f}%)', rotation=90, ha='right')

    plt.title('Distribution of Prediction Scores vs Actual November Result (±1σ and ±2σ)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.legend()
    plt.gca()
    return actual_result, lean_margin, std_dev, strong_margin


@app.cell
def _(actual_result, lean_margin, november_set, np, pd, strong_margin):
    # Recategorize predictions based on actual result
    november_set['prediction_sentiment'] = pd.cut(
        november_set['prediction_for_share'],
        bins=[-np.inf, actual_result - strong_margin, actual_result - lean_margin, actual_result + lean_margin, actual_result + strong_margin, np.inf],
        labels=['strongly_against', 'lean_against', 'swing', 'lean_for', 'strongly_for']
    )
    return


@app.cell
def _(november_set, pd):
    # First, calculate the actual ward and precinct level results
    ward_actual = (
        november_set
        .groupby('WARD')
        .agg(
            {
                'nov_for_share': 'mean'
            })
        .rename(
            columns={
                'nov_for_share': 'actual_ward_for_share'
            }
        ))

    precinct_actual = (
        november_set
        .groupby('PRECINCT_NAME')
        .agg(
            {
                'nov_for_share': 'mean'
            })
        .rename(
            columns={
                'nov_for_share': 'actual_precinct_for_share'
            }))

    age_range_actual = (
        november_set
        .groupby('AGE_RANGE')
        .agg(
            {
                'nov_for_share': 'mean'
            })
        .rename(
            columns={
                'nov_for_share': 'actual_age_range_for_share'
            }))

    # Calculate predicted results at ward and precinct level
    ward_predicted = (
        november_set
        .groupby('WARD')
        .agg(
            {
                'prediction_for_share': 'mean'
            })
        .rename(
            columns={
                'prediction_for_share': 'predicted_ward_for_share'
            }))

    precinct_predicted = (
        november_set
        .groupby('PRECINCT_NAME')
        .agg(
            {
                'prediction_for_share': 'mean'
            })
        .rename(
            columns={
                'prediction_for_share': 'predicted_precinct_for_share'
            }))

    age_range_predicted = (
        november_set
        .groupby('AGE_RANGE')
        .agg(
            {
                'prediction_for_share': 'mean'
            })
        .rename(
            columns={
                'prediction_for_share': 'predicted_age_range_for_share'
            }))

    # Combine actual and predicted results
    ward_comparison = pd.merge(
        ward_actual, ward_predicted, left_index=True, right_index=True)

    ward_comparison['difference'] = (
        ward_comparison['predicted_ward_for_share'] -
        ward_comparison['actual_ward_for_share'])


    precinct_comparison = pd.merge(
        precinct_actual, precinct_predicted, left_index=True, right_index=True)

    precinct_comparison['difference'] = (
        precinct_comparison['predicted_precinct_for_share'] - 
        precinct_comparison['actual_precinct_for_share'])

    age_range_comparison = pd.merge(
        age_range_actual, age_range_predicted, left_index=True, right_index=True)

    age_range_comparison['difference'] = (
        age_range_comparison['predicted_age_range_for_share'] - 
        age_range_comparison['actual_age_range_for_share'])
    return (
        age_range_actual,
        age_range_comparison,
        age_range_predicted,
        precinct_actual,
        precinct_comparison,
        precinct_predicted,
        ward_actual,
        ward_comparison,
        ward_predicted,
    )


@app.cell
def _(mo):
    mo.md(r"""## Ward Prediction vs Actual Comparison""")
    return


@app.cell
def _(mo, ward_comparison):
    # Print comparisons
    # print("\nWard-Level Comparison:")
    # print("=" * 50)
    mo.ui.table(ward_comparison.sort_values('difference', ascending=False).round(4))

    # # print("\nPrecinct-Level Comparison:")
    # # print("=" * 50)
    # mo.ui.table(precinct_comparison.sort_values('difference', ascending=False).round(4))

    # print("\nAge Range-Level Comparison:")
    # print("=" * 50)
    # mo.ui.table(age_range_comparison.sort_values('difference', ascending=False).round(4))
    return


@app.cell
def _(mo):
    mo.md(r"""## Precinct Prediction vs Actual Comparison""")
    return


@app.cell
def _(mo, precinct_comparison):
    mo.ui.table(precinct_comparison.sort_values('difference', ascending=False).round(4))
    return


@app.cell
def _(mo):
    mo.md(r"""## Age Range Prediction vs Actual Comparison""")
    return


@app.cell
def _(age_range_comparison, mo):
    mo.ui.table(age_range_comparison.sort_values('difference', ascending=False).round(4))
    return


@app.cell
def _(age_range_comparison, mo, precinct_comparison, ward_comparison):
    # Calculate summary statistics
    mo.md(f"""\nSummary Statistics:
    {'=' * 50}
    ## Ward-Level Differences  
    - Mean Absolute Error: {abs(ward_comparison['difference']).mean():.4f}  
    - Max Overprediction: {ward_comparison['difference'].max():.4f}  
    - Max Underprediction: {ward_comparison['difference'].min():.4f}  


    ## Precinct-Level Differences:  
    - Mean Absolute Error: {abs(precinct_comparison['difference']).mean():.4f}  
    - Max Overprediction: {precinct_comparison['difference'].max():.4f}  
    - Max Underprediction: {precinct_comparison['difference'].min():.4f}

    ## Age Range-Level Differences:  
    - Mean Absolute Error: {abs(age_range_comparison['difference']).mean():.4f}  
    - Max Overprediction: {age_range_comparison['difference'].max():.4f}  
    - Max Underprediction: {age_range_comparison['difference'].min():.4f}

    """)
    return


@app.cell
def _(m_data, november_set):
    november_results = november_set[november_set['VOTED_NOV_LEVY'] == True].groupby('prediction_sentiment')['VOTED_NOV_LEVY'].sum().reset_index()
    november_results['percent'] = (
        november_results['VOTED_NOV_LEVY'] / 
        m_data['VOTED_NOV_LEVY'].sum()).round(4)

    november_merge = november_set[['WARD', 'PRECINCT_NAME', 'AGE_RANGE','prediction_sentiment']].drop_duplicates()

    november_quant = november_set['prediction_for_share'].quantile([0.15, 0.4, 0.6, 0.85])
    return november_merge, november_quant, november_results


@app.cell
def _(pd):
    def calculate_votes_by_age_precinct(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total votes by age range and precinct, including for/against percentages.

        Args:
            data: DataFrame containing voter data with AGE_RANGE, PRECINCT_NAME, and vote columns

        Returns:
            DataFrame with total votes and percentages by age and precinct
        """
        # Group by age range and precinct
        age_precinct_stats = data.groupby(['AGE_RANGE', 'PRECINCT_NAME']).agg({
            'VOTED_NOV_LEVY': 'sum',  # Total votes
            'nov_for_share': 'mean',   # Average for percentage
            'nov_against_share': 'mean'  # Average against percentage
        }).reset_index()

        # Calculate total registered voters by age and precinct
        total_voters = data.groupby(['AGE_RANGE', 'PRECINCT_NAME']).size().reset_index(name='total_registered')

        # Merge the statistics
        results = age_precinct_stats.merge(total_voters, on=['AGE_RANGE', 'PRECINCT_NAME'])

        # Calculate percentages
        results['turnout_rate'] = (results['VOTED_NOV_LEVY'] / results['total_registered']).round(4)
        results['for_votes'] = (results['VOTED_NOV_LEVY'] * results['nov_for_share']).round(0)
        results['against_votes'] = (results['VOTED_NOV_LEVY'] * results['nov_against_share']).round(0)

        return results
    return (calculate_votes_by_age_precinct,)


@app.cell
def _(mo):
    mo.md(r"""## May Vote Result Predictions Based on Model""")
    return


@app.cell
def _(calculate_votes_by_age_precinct, m_data, mo):
    # Calculate and display the results
    votes_by_age_precinct = calculate_votes_by_age_precinct(m_data)
    # print("\nVotes by Age and Precinct:")
    # print("=" * 50)
    mo.ui.table(votes_by_age_precinct)
    return (votes_by_age_precinct,)


if __name__ == "__main__":
    app.run()
