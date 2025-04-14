import marimo

__generated_with = "0.12.8"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"# FCS Tax Levy - May 6 Election Breakdown")
    return


@app.cell
def _():
    import marimo as mo
    import fcs_may25.main as data
    import pandas as pd
    return data, mo, pd


@app.cell
def _(mo):
    mo.md(r"## November 2024 Results")
    return


@app.cell
def _(data, mo):
    mo.ui.table(data.election_results)
    return


@app.cell
def _(mo):
    mo.md(r"## November Breakdown: By Age Range")
    return


@app.cell
def _(data, mo):
    nov_turnout_age = data.november_turnout.groupby('AGE_RANGE')['SOS_VOTERID'].count().reset_index().rename(columns={'SOS_VOTERID': 'VOTES'})
    nov_turnout_age['PERCENTAGE'] = (nov_turnout_age['VOTES'] / nov_turnout_age['VOTES'].sum()).round(2)
    mo.ui.table(nov_turnout_age)
    return (nov_turnout_age,)


@app.cell
def _(mo):
    mo.md(r"## November Breakdown: By Ward")
    return


@app.cell
def _(data, mo):
    nov_turnout_ward = data.november_turnout.groupby('ward')['SOS_VOTERID'].count().reset_index().rename(columns={'SOS_VOTERID': 'VOTES'})
    nov_turnout_ward['PERCENTAGE'] = (nov_turnout_ward['VOTES'] / nov_turnout_ward['VOTES'].sum()).round(2)
    mo.ui.table(nov_turnout_ward)
    return (nov_turnout_ward,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## November Breakdown: By Precinct in Ward
        *Note: These totals may differ slightly from the FCS Levy results due to undervote.*
        """
    )
    return


@app.cell
def _(data, mo, pd):

    ward_precinct_turnout = pd.crosstab(
        index=[data.november_turnout['ward'], data.november_turnout['PRECINCT_NAME']],
        columns=[data.november_turnout['AGE_RANGE']],
        margins=True,
        margins_name='Total',
        normalize=True
    )
    mo.ui.table(ward_precinct_turnout)
    return (ward_precinct_turnout,)


@app.cell
def _(mo):
    mo.md(r"## May Turnout: By Precinct")
    return


@app.cell
def _(data, mo):
    may_vs_nov = data.november_turnout.groupby(['ward', 'PRECINCT_NAME'])[['VOTED_MAY_LEVY', 'VOTED_IN_NOV']].sum()
    may_vs_nov['MAY_PCT_OF_VOTES'] = (may_vs_nov['VOTED_MAY_LEVY'] / may_vs_nov['VOTED_MAY_LEVY'].sum()).round(2)
    may_vs_nov['NOV_PCT_OF_VOTES'] = (may_vs_nov['VOTED_IN_NOV'] / may_vs_nov['VOTED_IN_NOV'].sum()).round(2)
    mo.ui.table(may_vs_nov)
    return (may_vs_nov,)


@app.cell
def _(mo):
    mo.md(r"## May Turnout: By Ward")
    return


@app.cell
def _(may_vs_nov, mo):
    may_turnout_by_ward = may_vs_nov.groupby('ward')[['VOTED_MAY_LEVY', 'VOTED_IN_NOV', 'MAY_PCT_OF_VOTES', 'NOV_PCT_OF_VOTES']].sum()
    mo.ui.table(may_turnout_by_ward)
    return (may_turnout_by_ward,)


@app.cell
def _(mo):
    mo.md(r"## May Turnout: By Age Range")
    return


@app.cell
def _(data, mo):
    may_by_age_range = data.november_turnout.groupby('AGE_RANGE')[['VOTED_MAY_LEVY', 'VOTED_IN_NOV']].sum().reset_index()
    may_by_age_range['MAY_PCT_OF_VOTES'] = (may_by_age_range['VOTED_MAY_LEVY'] / may_by_age_range['VOTED_MAY_LEVY'].sum()).round(2)
    may_by_age_range['NOV_PCT_OF_VOTES'] = (may_by_age_range['VOTED_IN_NOV'] / may_by_age_range['VOTED_IN_NOV'].sum()).round(2)
    mo.ui.table(may_by_age_range)
    return (may_by_age_range,)


@app.cell
def _(mo):
    mo.md(r"## May Turnout: Predicted Votes By Ward")
    return


@app.cell
def _(data, mo):
    predicted_results = data.prediction_and_results.copy()
    predicted_results = predicted_results.rename(
        columns={
            'total_votes': 'may_total_votes',
            'prediction_against': 'may_prediction_against',
            'prediction_for': 'may_prediction_for',
            'prediction_swing': 'may_prediction_swing',
            'pct_for': 'may_pct_for',
            'pct_against': 'may_pct_against',
            'pct_swing': 'may_pct_swing',
            'for': 'for_in_nov', 
            'against': 'against_in_nov', 
            'total': 'total_in_nov'})
    p_cols = list(predicted_results.columns)
    predicted_results = predicted_results[[p_cols.pop(p_cols.index('ward'))] + p_cols]
    mo.ui.table(predicted_results)
    return p_cols, predicted_results


@app.cell
def _(mo):
    mo.md(r"## Comparisons")
    return


@app.cell
def _(mo):
    mo.md(r"### Vote Makeup By Ward: November")
    return


@app.cell
def _(mo, nov_turnout_ward):
    nov_ward_pie = nov_turnout_ward['PERCENTAGE'].plot(
        kind='pie', 
        autopct='%1.1f%%', 
        labels=nov_turnout_ward['ward'],
        legend=False,
        ylabel='',
    )
    mo.mpl.interactive(nov_ward_pie)
    return (nov_ward_pie,)


@app.cell
def _(mo):
    mo.md(r"### Vote Makeup By Ward: May")
    return


@app.cell
def _(may_turnout_by_ward, mo):
    may_ward_pie = may_turnout_by_ward['MAY_PCT_OF_VOTES'].plot(kind='pie', autopct='%1.1f%%', labels=may_turnout_by_ward['ward'], legend=False, ylabel='')
    mo.mpl.interactive(may_ward_pie)

    return (may_ward_pie,)


@app.cell
def _(mo):
    mo.md(r"### Vote Makeup By Age Range: November")
    return


@app.cell
def _(mo, nov_turnout_age):
    nov_age_range_pie = nov_turnout_age['PERCENTAGE'].plot(kind='pie', autopct='%1.1f%%', labels=nov_turnout_age['AGE_RANGE'], legend=False, ylabel='')
    mo.mpl.interactive(nov_age_range_pie)
    return (nov_age_range_pie,)


@app.cell
def _(mo):
    mo.md(r"### Vote Makeup By Age Range: May")
    return


@app.cell
def _(may_by_age_range, mo):
    may_age_range_pie = may_by_age_range['MAY_PCT_OF_VOTES'].plot(kind='pie', autopct='%1.1f%%', labels=may_by_age_range['AGE_RANGE'], legend=False, ylabel='')
    mo.mpl.interactive(may_age_range_pie)
    return (may_age_range_pie,)


@app.cell
def _(mo):
    mo.md(r"### Vote Makeup By Age Range: Comparison")
    return


@app.cell
def _(may_by_age_range, mo):
    # Make a combined bar chart of the vote makeup by age range for both November and May side by side
    may_age_range_bar = may_by_age_range.plot(kind='bar', x='AGE_RANGE', y=['MAY_PCT_OF_VOTES', 'NOV_PCT_OF_VOTES'], ylabel='Percentage of Votes', legend=True)
    mo.mpl.interactive(may_age_range_bar)

    return (may_age_range_bar,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
