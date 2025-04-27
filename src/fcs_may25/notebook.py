import marimo

__generated_with = "0.12.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from icecream import ic
    import numpy as np
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
    from sklearn.compose import ColumnTransformer
    import pandas as pd

    from data_loader import FindlayVoterFile, FindlayVoterFileConfig as vf_config, NovemberResultsColumns as nov_results
    from config import FindlayLinearModelFeatureLists as ml_cat
    from prediction import (
        FindlayPredictionModel,
        test_max_iter_impact,
        plot_regression_results,
        plot_feature_importance
    )
    from main import calculate_turnout
    return (
        ColumnTransformer,
        FindlayPredictionModel,
        FindlayVoterFile,
        LinearRegression,
        OneHotEncoder,
        StandardScaler,
        calculate_turnout,
        ic,
        mean_squared_error,
        ml_cat,
        mo,
        nov_results,
        np,
        pd,
        plot_feature_importance,
        plot_regression_results,
        r2_score,
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
def _(mo):
    mo.md(
        r"""
        ## Model Preparation
        ### City-Wide Prep
        1. Age Range/Ward & Age Range/PrecinctNormalization
    
        ### Individual District Prep
        1. Age Range Normalization within each Ward/Precinct
        """
    )
    return


@app.cell
def _(m_data, pd):
    age_ward_city = pd.crosstab(m_data['AGE_RANGE'], m_data['WARD'], margins=True, normalize='all')
    age_precinct_city = pd.crosstab(m_data['AGE_RANGE'], m_data['PRECINCT_NAME'], margins=True, normalize='all')

    # 2. Get the age distribution within each ward/precinct
    age_ward_within = pd.crosstab(m_data['AGE_RANGE'], m_data['WARD'], margins=True, normalize='columns')
    age_precinct_within = pd.crosstab(m_data['AGE_RANGE'], m_data['PRECINCT_NAME'], margins=True, normalize='columns')
    return (
        age_precinct_city,
        age_precinct_within,
        age_ward_city,
        age_ward_within,
    )


@app.cell
def _(mo):
    mo.md(r"## Calculate Primary Turnout Stats")
    return


@app.cell
def _(calculate_turnout, m_data, vf_config):
    primary_elections = list(vf_config.PRIMARY_COLUMNS.keys())
    general_elections = list(vf_config.GENERAL_COLUMNS.keys())
    m_data[primary_elections] = m_data[primary_elections].astype(bool)
    m_data[general_elections] = m_data[general_elections].astype(bool)

    m_data1 = m_data.merge(calculate_turnout(m_data, primary_elections, by_column='AGE_RANGE', district_level='WARD'), on=['AGE_RANGE', 'WARD'], how='left')
    m_data2 = m_data1.merge(calculate_turnout(m_data1, general_elections, by_column='AGE_RANGE', district_level='WARD'), on=['AGE_RANGE', 'WARD'], how='left')

    m_data3 = m_data2.merge(calculate_turnout(m_data2, primary_elections, by_column='AGE_RANGE', district_level='PRECINCT_NAME'), on=['AGE_RANGE', 'PRECINCT_NAME'], how='left')
    m_data4 = m_data3.merge(calculate_turnout(m_data3, general_elections, by_column='AGE_RANGE', district_level='PRECINCT_NAME'), on=['AGE_RANGE', 'PRECINCT_NAME'], how='left')

    m_data5 = m_data4.merge(calculate_turnout(m_data4, primary_elections, by_column='PARTY_AFFILIATION', district_level='WARD'), on=['PARTY_AFFILIATION', 'WARD'], how='left')
    m_data6 = m_data5.merge(calculate_turnout(m_data5, general_elections, by_column='PARTY_AFFILIATION', district_level='WARD'), on=['PARTY_AFFILIATION', 'WARD'], how='left')

    m_data7 = m_data6.merge(calculate_turnout(m_data6, primary_elections, by_column='PARTY_AFFILIATION', district_level='PRECINCT_NAME'), on=['PARTY_AFFILIATION', 'PRECINCT_NAME'], how='left')
    m_data8 = m_data7.merge(calculate_turnout(m_data7, general_elections, by_column='PARTY_AFFILIATION', district_level='PRECINCT_NAME'), on=['PARTY_AFFILIATION', 'PRECINCT_NAME'], how='left')

    merged_data = m_data8
    return (
        general_elections,
        m_data1,
        m_data2,
        m_data3,
        m_data4,
        m_data5,
        m_data6,
        m_data7,
        m_data8,
        merged_data,
        primary_elections,
    )


@app.cell
def _(mo):
    mo.md(r"## Setup Catigorical Features")
    return


@app.cell
def _(merged_data, ml_cat, pd):
    # Category Features
    merged_data[age_range_cat := 'AGE_RANGE_CAT'] = pd.Categorical(merged_data['AGE_RANGE'], categories=sorted(merged_data['AGE_RANGE'].unique()), ordered=True)
    merged_data[party_cat := 'PARTY_CAT'] = pd.Categorical(merged_data['PARTY_AFFILIATION'], categories=['D', 'I', 'R'], ordered=True)
    ml_cat.category_features.extend([age_range_cat, party_cat])
    return age_range_cat, party_cat


@app.cell
def _(mo):
    mo.md(r"## Setup Interaction Features")
    return


@app.cell
def _(merged_data, ml_cat, pd):
    merged_data[age_ward := 'AGE_WARD'] = merged_data['AGE_RANGE'].astype(str) + '-' + merged_data['WARD'].astype(str)
    merged_data[age_precinct := 'AGE_PRECINCT'] = merged_data['AGE_RANGE'].astype(str) + '-' + merged_data['PRECINCT_NAME'].astype(str)
    merged_data[age_party := 'AGE_PARTY'] = merged_data['AGE_RANGE'].astype(str) + '-' + merged_data['PARTY_AFFILIATION'].astype(str)
    # Score Features
    merged_data[p_score_last4_cat := 'P_SCORE_LAST4_CAT'] = pd.cut(
        merged_data['P_SCORE'],
        bins=5,
        labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    ).astype(int)
    merged_data[g_score_last4_cat := 'G_SCORE_LAST4_CAT'] = pd.cut(
        merged_data['G_SCORE'],
        bins=5,
        labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    ).astype(int)
    merged_data[p_score_all_cat := 'P_SCORE_ALL_CAT'] = pd.cut(
        merged_data['P_SCORE_ALL'],
        bins=5,
        labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    ).astype(int)
    merged_data[g_score_all_cat := 'G_SCORE_ALL_CAT'] = pd.cut(
        merged_data['G_SCORE_ALL'],
        bins=5,
        labels=[0, 1, 2, 3, 4] # strongly against, lean against, lean for, strongly for
    ).astype(int)

    merged_data[p_score_last4_ward := 'P_SCORE_LAST4_WARD'] = merged_data[p_score_last4_cat].astype(str) + '-' + merged_data[age_ward].astype(str)
    merged_data[g_score_last4_ward := 'G_SCORE_LAST4_WARD'] = merged_data[g_score_last4_cat].astype(str) + '-' + merged_data[age_ward].astype(str)
    merged_data[p_score_all_ward := 'P_SCORE_ALL_WARD'] = merged_data[p_score_all_cat].astype(str) + '-' + merged_data[age_ward].astype(str)
    merged_data[g_score_all_ward := 'G_SCORE_ALL_WARD'] = merged_data[g_score_all_cat].astype(str) + '-' + merged_data[age_ward].astype(str)
    merged_data[p_score_last4_age_ward_precinct := 'P_SCORE_LAST4_AGE_WARD_PRECINCT'] = merged_data[p_score_last4_cat].astype(str) + '-' + merged_data[age_ward].astype(str) + '-' + merged_data[age_precinct].astype(str)
    merged_data[g_score_last4_age_ward_precinct := 'G_SCORE_LAST4_AGE_WARD_PRECINCT'] = merged_data[g_score_last4_cat].astype(str) + '-' + merged_data[age_ward].astype(str) + '-' + merged_data[age_precinct].astype(str)
    merged_data[p_score_all_age_ward_precinct := 'P_SCORE_ALL_AGE_WARD_PRECINCT'] = merged_data[p_score_all_cat].astype(str) + '-' + merged_data[age_ward].astype(str) + '-' + merged_data[age_precinct].astype(str)
    merged_data[g_score_all_age_ward_precinct := 'G_SCORE_ALL_AGE_WARD_PRECINCT'] = merged_data[g_score_all_cat].astype(str) + '-' + merged_data[age_ward].astype(str) + '-' + merged_data[age_precinct].astype(str)
    ml_cat.interaction_features.extend([
        age_ward,
        age_precinct,
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
    return (
        age_party,
        age_precinct,
        age_ward,
        g_score_all_age_ward_precinct,
        g_score_all_cat,
        g_score_all_ward,
        g_score_last4_age_ward_precinct,
        g_score_last4_cat,
        g_score_last4_ward,
        p_score_all_age_ward_precinct,
        p_score_all_cat,
        p_score_all_ward,
        p_score_last4_age_ward_precinct,
        p_score_last4_cat,
        p_score_last4_ward,
    )


@app.cell
def _(mo):
    mo.md(r"## Linear Regression Model")
    return


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    merged_data,
    ml_cat,
    pd,
    plot_feature_importance,
    plot_regression_results,
    train_test_split,
):
    y_pseudo = merged_data['nov_for'] / merged_data['total']

    # Convert categorical columns to dummy variables
    categorical_dummies = pd.get_dummies(merged_data[ml_cat.category_features + ml_cat.interaction_features], 
                                      drop_first=True, # Drop first category to avoid multicollinearity
                                      dtype=float)      # Convert to float for the model

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = pd.DataFrame(
        scaler.fit_transform(merged_data[ml_cat.numerical_features]),
        columns=ml_cat.numerical_features,
        index=merged_data.index
    )

    # Combine all features
    X = pd.concat([numerical_features, categorical_dummies], axis=1)
    y = y_pseudo  # Using the same target variable as before

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Now plot the results
    plot_regression_results(y_test, y_pred, 'Linear Regression')
    plot_feature_importance(model, X.columns)
    return (
        X,
        X_test,
        X_train,
        categorical_dummies,
        model,
        numerical_features,
        scaler,
        y,
        y_pred,
        y_pseudo,
        y_test,
        y_train,
    )


if __name__ == "__main__":
    app.run()
