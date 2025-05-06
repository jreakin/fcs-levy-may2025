import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from icecream import ic
    import seaborn as sns
    import pandas as pd
    from functools import partial
    import matplotlib.pyplot as plt
    from typing import Optional

    from data_loader import FindlayVoterFile, FindlayVoterFileConfig as vf_config, NovemberResultsColumns as nov_results
    from config import FindlayLinearModelFeatureLists as ml_cat
    import main as m
    return (
        FindlayVoterFile,
        Optional,
        ic,
        m,
        ml_cat,
        mo,
        nov_results,
        partial,
        pd,
        plt,
        sns,
        vf_config,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Findlay City Schools Tax Levy
        # Turnout Report
        <center> Prepared by: John R. Eakin,  Abstract Data </center>
        """
    )
    return


@app.cell
def _(Optional, partial, pd, plt):
    # Function Setup
    nov_may_voters_rename = {'nov_voters': 'November', 'may_voters': 'May'}
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
            title=title,
            ylabel=""
        )

    new_subplot = partial(plt.subplots, 1, 2, figsize=(15,7))
    return create_pie_chart, new_subplot, nov_may_voters_rename


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ## <center> Turnout Comparisons </center>
        ---
        Observations:  

        - Voters 45 and older make up approximately 41.6% of the registered voters, not quite 60% as previously stated. However, these voters still form the core power base for passage of the levy due to their reliability.
        - The 25-34 age range is substantial at 18.9%, and along with the 35-44 group at 16.5%, these middle-aged voters represent over 35% of registered voters but remain inconsistent performers.
        - The pathway to victory for the Levy runs straight through older homeowners, with particular focus on 55-64 year olds who constitute 15.7% of registered voters - the third largest age demographic.
        <br>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### <center> City-Wide Age Ranges for Registered Voters </center>
        <br>
        Observations: 

        - The 25-34 bracket shows approximately 5,000 voters - our largest age demographic with tons of potential but historically unreliable turnout.
        - The 35-54 age brackets give us another 8,000+ voters who are the swing vote on school funding - parents with kids in the system who calculate their tax burden against educational benefits.
        - 55+ voters are roughly 10,000 strong (not 11,000) and they vote religiously, with the 55-64 group being particularly numerous at around 4,000 registered voters.
        - When turnout drops in May elections, these seniors become even more dominant, as their representation nearly doubles in the electorate.
        """
    )
    return


@app.cell
def _(create_pie_chart, m, new_subplot, plt):
    fig0, (ax01, ax02) = new_subplot()
    create_pie_chart(m.m_data['AGE_RANGE'], title='All Voters', ax=ax01)
    m.m_data['AGE_RANGE'].value_counts().sort_index().plot(
        kind='bar', 
        ylabel='Count',
        xlabel='Age Range',
        title='City-Wide Voter Count By Age Range',
        ax=ax02)
    plt.gca()
    return ax01, ax02, fig0


@app.cell
def _(mo):
    mo.md(r"""### <center> Age Range Percentage By Ward </center>""")
    return


@app.cell
def _(m, mo, pd):
    age_range_ct = pd.crosstab(
        index=m.m_data['WARD'].str.title(),
        columns=m.m_data['AGE_RANGE'],
        margins=True,
        normalize='index'
    ).round(4)
    mo.ui.table(age_range_ct, page_size=12)
    return (age_range_ct,)


@app.cell
def _(mo):
    mo.md(r"""## <center> May Early Vote </center>""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""### <center> Early Vote: By Method </center>""")
    return


@app.cell
def _(m, plt):
    vote_methods = m.voterfile.current_votes
    merge_methods = vote_methods.merge(m.m_data[['SOS_VOTERID', 'AGE_RANGE', 'WARD']], right_on='SOS_VOTERID', left_on='STATE ID#')

    city_by_method = (
        merge_methods['Vote Method']
        .value_counts(normalize=True)
        .plot(
            kind='pie', 
            autopct='%1.1f%%',
            ylabel='')
    )

    plt.show()
    return city_by_method, merge_methods, vote_methods


@app.cell
def _(merge_methods, new_subplot, pd, plt):
    fig00, (ax001, ax002) = new_subplot()
    age_ct = (
        pd.crosstab(
        index=merge_methods['AGE_RANGE'],
        columns=merge_methods['Vote Method'])
        .reset_index()
        .rename(columns={'AGE_RANGE': 'Age Range'})
        .plot(
            kind='bar', 
            y=['In-Person', 'Mail'], 
            x='Age Range',
            ax=ax001,
            title='Early Vote By Age Range'
        )
    )

    ward_ct = (
        pd.crosstab(
        index=merge_methods['WARD'].str.title(),
        columns=merge_methods['Vote Method'])
        .reset_index()
        .plot(
            kind='bar', 
            y=['In-Person', 'Mail'], 
            x='WARD',
            ax=ax002,
            title="Early Vote By Ward"
        )
    )
    plt.tight_layout()
    plt.gcf()
    return age_ct, ax001, ax002, fig00, ward_ct


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ### <center> November vs May: Turnout By Age Range </center>
        This is the whole ballgame right here. 

        November turnout is always higher across the board – that's Politics 101. But look at what happens in May: the 18-24 bracket virtually disappears, and 25-44 participation craters. Meanwhile, those 65+ voters? They're still showing up at nearly 30% turnout rates! 

        Observations: 

        - The battle for May will be fought and won in the retirement communities, church groups, and senior centers.
        - Every voter under 45 who actually shows up in May is worth about three November voters in terms of electoral impact.
        <br>
        """
    )
    return


@app.cell
def _(create_pie_chart, m, new_subplot, plt):
    fig1, (ax1, ax2) = new_subplot()

    create_pie_chart(m.november['AGE_RANGE'], title='November', ax=ax1)
    create_pie_chart(m.may['AGE_RANGE'], title='May', ax=ax2)
    plt.gca()
    return ax1, ax2, fig1


@app.cell
def _(m, plt):
    m.merge_ages.rename(
        columns={
            'nov_count': 'November', 
            'may_count': 'May'}).plot(
        kind='bar', 
        y=['November', 'May'],
        x='AGE_RANGE',
        xlabel='Age Range',
        ylabel='Percent',
        title='Turnout Percentage By Age Range',
        figsize=(12, 6)
    )

    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ### <center> Turnout By Ward </center>
        Ward-level analysis separates the professionals from the amateurs. 

        - Ward 5 delivers 16.7% of November voters but drops to only 15.8% in May – that's leakage we need to address.
        - Meanwhile, Ward 2 jumps from 12.8% to 21.8% of the electorate in May! That's not an accident – that's where your seniors live. This ward's weight could potentially double in the May election.
        - Ward 3 and 7 show strong "for" percentages in both elections, making them your base precincts, with Ward 3 consistently delivering a 15.5% share of voters.
        - Liberty S and Ward 6 show wildly different behavior between elections – Liberty S drops significantly while Ward 6 plummets from 8.1% to 4.2%.
        <br>
        """
    )
    return


@app.cell
def _(create_pie_chart, m, new_subplot, plt):
    fig2, (ax3, ax4) = new_subplot()

    create_pie_chart(m.november['WARD'].apply(lambda x: x.title()), title='November', ax=ax3)
    create_pie_chart(m.may['WARD'].apply(lambda x: x.title()), title='May', ax=ax4)
    plt.gca()
    return ax3, ax4, fig2


@app.cell
def _(m, new_subplot, nov_may_voters_rename, plt):
    fig, (ax5, ax6) = new_subplot()

    m.merge_by_ward.rename(columns=nov_may_voters_rename).plot(
        kind='bar',
        y=['November', 'May'],
        title='Nov vs. May Turnout By Ward',
        ylabel='Percent',
        figsize=(12, 6)
    )
    plt.gcf()
    return ax5, ax6, fig


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ## <center> Prediction Modeling </center>
        ---
        Our modeling isn't just educated guesswork– it's tactical intelligence. The actual November result was 47.4% "for" votes and 52.6% "against" votes as shown in the outcome chart.

        The distribution curves indicate between 45-50% support in a typical election, with our May model predicting 48.06% support - slightly better than November but still short of passage.

        May isn't typical – it's an opportunity if we can enhance turnout in pro-levy areas. The ward-level predictions identify exactly where the levy's ground game needs to focus.
        <br>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### <center> Prediction Density </center>
        <br>
        """
    )
    return


@app.cell
def _(m, plt, sns):
    # Create KDE plot for prediction distributions
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=m.november_set['P_for'], 
        label='Prediction Share', 
        alpha=0.7
    )
    return


@app.cell
def _(m, plt):
    plt.figure(figsize=(12, 6))
    # Add vertical line for actual result
    plt.axvline(x=m.actual_result, color='red', linestyle='-', alpha=0.8, label='Actual November Result (46.40%)')

    # Add threshold lines based on margins from actual result
    plt.axvline(x=m.actual_result - m.strong_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result - m.lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result + m.lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result + m.strong_margin, color='gray', linestyle='--', alpha=0.5)
    y_min, y_max = plt.ylim()
    y_center = (y_min + y_max) / 2 

    # Add labels
    plt.text(m.actual_result - m.strong_margin, y_center, f'\nStrongly Against\n(<{(m.actual_result-m.strong_margin)*100:.1f}%)', rotation=90, ha='right', va='center')
    plt.text(m.actual_result - m.lean_margin, y_center, f'\nLean Against\n(<{(m.actual_result-m.lean_margin)*100:.1f}%)', rotation=90, ha='right', va='center')
    plt.text(m.actual_result + m.lean_margin, y_center, f'\nLean For\n(>{(m.actual_result+m.lean_margin)*100:.1f}%)', rotation=90, ha='right', va='center')
    plt.text(m.actual_result + m.strong_margin, y_center, f'\nStrongly For\n(>{(m.actual_result+m.strong_margin)*100:.1f}%)', rotation=90, ha='right', va='center')

    plt.title('Distribution of Prediction Scores vs Actual November Result (±1σ and ±2σ)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.legend()
    plt.gca()
    return y_center, y_max, y_min


@app.cell
def _(m, plt, sns):
    # Add visualization of the new distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=m.november_set, x='P_for', bins=50)
    plt.axvline(x=m.actual_result, color='red', linestyle='-', label='Actual November Result (46.40%)')
    plt.axvline(x=m.actual_result - m.strong_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result - m.lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result + m.lean_margin, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=m.actual_result + m.strong_margin, color='gray', linestyle='--', alpha=0.5)
    plt.title('Distribution of Adjusted Prediction Scores')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ### <center> Ward Prediction vs Actual Comparison </center>
        Observations: 

        - Ward 6 underperformed model expectations by 4.5 points.
        - The overall mean absolute error of 0.0321 at the ward level means our model is generally accurate, but in an election projected to be decided by under 4 percentage points (48.06% vs 51.94%), even small turnout variations could swing the result.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <br>
        ### <center> November Precinct Prediction vs Actual Comparison </center>

        - Findlay 6A underperformed by a whopping 12.7 points – someone needs to explain that disaster. Liberty SE outperformed by 14.7 points – what did they do right?
        - When I see precinct-level variations this dramatic, it tells me the ground game is inconsistent. The mean absolute error of 0.0464 at the precinct level indicates we need better coordination across neighborhoods.
        - The ward-level charts show stark differences between November and May predicted votes, with Findlay 5 showing the biggest drop in raw vote numbers from November to May.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## <center> Breakdown of Age Range Vote Predictions </center>""")
    return


@app.cell
def _(m, plt, sns):
    plt.figure(figsize=(12, 8))
    sns.heatmap(m.age_ward_predictions, annot=True, cmap='RdYlBu', center=0.5)
    plt.title('Average Predictions by Age Range and Ward')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(m, plt, sns):
    plt.figure(figsize=(24, 12))
    sns.heatmap(m.age_precinct_predictions, annot=True, cmap='RdYlBu', center=0.5)
    plt.title('Average Predictions by Age Range and Precinct')
    plt.tight_layout()
    plt.gcf()
    return


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
    mo.md(
        r"""
        <br>
        ### May Vote Result Predictions Based on Model

        Pro-Tax Levy battle plan for May: 

        - Overall turnout will be half that of November, but the senior citizen army will show up in force. Their turnout rates will approach 30% while the under-35 crowd will barely break 5%.
        - That means every single 65+ voter could be worth up to SIX young voters in electoral impact.
        - The model shows the levy is on a knife's edge for passage – every vote will count.
        - Liberty S and Precinct 2C are the strongest performing areas and need to continue turning out high.
        - Precinct 6A and Marion N are areas where pro-Levy needs to stop the bleeding.

        In May elections, it's not who supports the levy – it's who actually shows up that determines the outcome.
        """
    )
    return


@app.cell
def _(calculate_votes_by_age_precinct, m):
    votes_by_age_precinct = calculate_votes_by_age_precinct(m.m_data)
    votes_by_age_precinct['turnout_rate'] = votes_by_age_precinct['turnout_rate'].astype(float).round(4)
    votes_by_age_precinct[['for_votes', 'against_votes']] = votes_by_age_precinct[['for_votes', 'against_votes']].round().astype(int)
    votes_by_age_precinct[['for_votes', 'against_votes']].rename(
        columns={
            'for_votes': 'For', 
            'against_votes': 'Against'}
    ).sum().plot(
        kind='pie', 
        autopct='%1.1f%%',
        title='Outcome Based November Precinct/Ward Results'
    )
    return (votes_by_age_precinct,)


@app.cell
def _(m, plt):
    (
        m.merged_results
        .rename(
            columns={
                'nov_ward_for_share': 'November', 
                'may_est_percent_for': 'May Predicted'
            })
        .plot(
            kind='bar',
            y=['November', 'May Predicted'],
            x='WARD',
            title='Predicted "for" Percentage by Ward Based on Estimated Turnout'
        )
    )
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""### <center> November vs. May Predictions: By Ward </center>""")
    return


@app.cell
def _(m, new_subplot, plt):
    fig3, (ax7, ax8) = new_subplot()
    (
        m.merged_results
        .rename(
            columns={
                'nov_ward_against_count': 'November Against', 
                'nov_ward_for_count': 'November For'})
        .plot(
            kind='bar',
            y=['November Against', 'November For'],
            x='WARD',
            ax=ax7
        )
    )

    (
        m.merged_results
        .rename(
            columns={
                'may_est_total_against': 'May Predicted Against', 
                'may_est_total_for': 'May Predicted For'})
        .plot(
            kind='bar',
            y=['May Predicted Against', 'May Predicted For'],
            x='WARD',
            ax=ax8
        )
    )
    plt.tight_layout()
    plt.gcf()
    return ax7, ax8, fig3


@app.cell
def _(m, mo):
    total_votes = (
        (_for := m.merged_results['may_est_total_for'].sum()) + 
        (_against := m.merged_results['may_est_total_against'].sum())
    )

    mo.md(f"""
    Model Prediction **For**: {_for.sum()} ({round(_for / total_votes, 4):.2%})  
    Model Prediction **Against**: {_against.sum()} ({round(_against / total_votes, 4):.2%})

    """
    )
    return (total_votes,)


if __name__ == "__main__":
    app.run()
