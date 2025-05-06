import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from fcs_may25.config import FilePaths
from tqdm import tqdm


@dataclass
class MonteCarloConfig:
    n_simulations: int = 100
    confidence_level: float = 0.95
    random_seed: int = 42
    turnout_std: float = 0.15
    sentiment_std: float = 0.2

class MonteCarloVoterSimulation:
    def __init__(self, data: pd.DataFrame, config: MonteCarloConfig = MonteCarloConfig()):
        self.data = data
        self.config = config
        self.simulation_results = []
        self.voter_predictions = pd.DataFrame()  # Initialize as empty DataFrame
        np.random.seed(config.random_seed)
        
        # Calculate historical May vs November and May vs Primary turnout ratios
        self.may_nov_ratio = self.calculate_may_november_ratio()
        self.may_primary_ratio = self.calculate_may_primary_ratio()
        
    def calculate_may_november_ratio(self) -> dict:
        """Calculate historical ratios between May and November turnout"""
        turnout_ratios = {
            'overall': 0.65,  # Default ratio if no historical data
            'by_age': {},
            'by_party': {},
            'by_precinct': {}
        }
        
        # Calculate age group ratios if historical data exists
        if 'AGE_RANGE' in self.data.columns:
            age_groups = self.data.groupby('AGE_RANGE').agg({
                'VOTED_MAY_LEVY': 'mean',
                'VOTED_NOV_LEVY': 'mean'
            })
            turnout_ratios['by_age'] = (age_groups['VOTED_MAY_LEVY'] / 
                                      age_groups['VOTED_NOV_LEVY']).fillna(0.65).to_dict()
        
        # Calculate party ratios if historical data exists
        if 'PARTY_CAT' in self.data.columns:
            party_groups = self.data.groupby('PARTY_CAT').agg({
                'VOTED_MAY_LEVY': 'mean',
                'VOTED_NOV_LEVY': 'mean'
            })
            turnout_ratios['by_party'] = (party_groups['VOTED_MAY_LEVY'] / 
                                        party_groups['VOTED_NOV_LEVY']).fillna(0.65).to_dict()
        
        # Calculate precinct ratios if historical data exists
        if 'PRECINCT_NAME' in self.data.columns:
            precinct_groups = self.data.groupby('PRECINCT_NAME').agg({
                'VOTED_MAY_LEVY': 'mean',
                'VOTED_NOV_LEVY': 'mean'
            })
            turnout_ratios['by_precinct'] = (precinct_groups['VOTED_MAY_LEVY'] / 
                                           precinct_groups['VOTED_NOV_LEVY']).fillna(0.65).to_dict()
        
        return turnout_ratios
    
    def calculate_may_primary_ratio(self) -> dict:
        """Calculate historical ratios between May and Primary turnout"""
        turnout_ratios = {
            'overall': 0.85,  # Default ratio if no historical data
            'by_age': {},
            'by_party': {},
            'by_precinct': {}
        }
        
        # Calculate age group ratios
        if 'AGE_RANGE' in self.data.columns:
            age_groups = self.data.groupby('AGE_RANGE').agg({
                'VOTED_MAY_LEVY': 'mean',
                'P_SCORE': 'mean'  # Primary score represents primary election participation
            })
            turnout_ratios['by_age'] = (age_groups['VOTED_MAY_LEVY'] / 
                                      age_groups['P_SCORE']).fillna(0.85).to_dict()
        
        # Calculate party ratios
        if 'PARTY_CAT' in self.data.columns:
            party_groups = self.data.groupby('PARTY_CAT').agg({
                'VOTED_MAY_LEVY': 'mean',
                'P_SCORE': 'mean'
            })
            turnout_ratios['by_party'] = (party_groups['VOTED_MAY_LEVY'] / 
                                        party_groups['P_SCORE']).fillna(0.85).to_dict()
        
        # Calculate precinct ratios
        if 'PRECINCT_NAME' in self.data.columns:
            precinct_groups = self.data.groupby('PRECINCT_NAME').agg({
                'VOTED_MAY_LEVY': 'mean',
                'P_SCORE': 'mean'
            })
            turnout_ratios['by_precinct'] = (precinct_groups['VOTED_MAY_LEVY'] / 
                                           precinct_groups['P_SCORE']).fillna(0.85).to_dict()
        
        return turnout_ratios
    
    def simulate_may_turnout(self, row: pd.Series) -> float:
        """Simulate May turnout probability using both November and Primary data."""
        # Convert P_SCORE to probability
        p_score_prob = row['P_SCORE'] / 100.0
        
        # Calculate ward-level primary turnout ratio
        ward_data = self.data[self.data['WARD'] == row['WARD']]
        ward_primary_ratio = (ward_data['P_SCORE'].mean() / ward_data['nov_for_share'].mean()) if len(ward_data) > 0 else 0.5
        
        # Calculate precinct-level primary turnout ratio
        precinct_data = self.data[
            (self.data['PRECINCT_NAME'] == row['PRECINCT_NAME']) & 
            (self.data['WARD'] == row['WARD'])
        ]
        precinct_primary_ratio = (precinct_data['P_SCORE'].mean() / precinct_data['nov_for_share'].mean()) if len(precinct_data) > 0 else 0.5
        
        # Calculate weighted probability
        # 70% weight to primary data, 30% to November data
        weighted_prob = (
            0.7 * p_score_prob +  # Primary data
            0.3 * (row['nov_for_share'] * ward_primary_ratio)  # November data adjusted by ward ratio
        ) / 1.0  # Normalize weights
        
        # Add random variation
        variation = np.random.normal(0, self.config.turnout_std)
        final_prob = np.clip(weighted_prob + variation, 0, 1)
        
        return final_prob
    
    def simulate_sentiment(self, vote_prediction: str) -> float:
        """Simulate voter sentiment with random variation based on vote prediction"""
        # Base probabilities for each prediction category
        base_probs = {
            'strongly_for': 0.8,
            'swing_for': 0.65,
            'lean_for': 0.55,
            'swing': 0.5,
            'lean_against': 0.45,
            'swing_against': 0.35,
            'strongly_against': 0.2
        }
        
        # Get base probability for the prediction
        base_prob = base_probs.get(vote_prediction, 0.5)
        
        # Add random variation
        final_prob = np.clip(
            np.random.normal(base_prob, self.config.sentiment_std),
            0, 1
        )
        
        return final_prob
    
    def validate_against_results(self):
        """Validate Monte Carlo predictions against actual election results"""
        if not self.simulation_results:
            self.run_simulation()
            
        # Calculate actual results
        actual_for_share = self.data['nov_for_share'].mean()
        actual_turnout = self.data['VOTED_MAY_LEVY'].mean()
        
        # Calculate prediction metrics
        predicted_for_share = np.mean([r['vote_prob'] for r in self.simulation_results])
        predicted_turnout = np.mean([r['turnout'] for r in self.simulation_results])
        
        # Calculate confidence intervals
        ci = self.get_confidence_intervals()
        
        # Print validation results
        print("\nMonte Carlo Validation Results:")
        print("=" * 50)
        print(f"Actual For Share: {actual_for_share:.4f}")
        print(f"Predicted For Share: {predicted_for_share:.4f}")
        print(f"For Share 95% CI: [{ci['for_share'][0]:.4f}, {ci['for_share'][1]:.4f}]")
        print(f"For Share Error: {abs(actual_for_share - predicted_for_share):.4f}")
        print(f"\nActual Turnout: {actual_turnout:.4f}")
        print(f"Predicted Turnout: {predicted_turnout:.4f}")
        print(f"Turnout 95% CI: [{ci['turnout'][0]:.4f}, {ci['turnout'][1]:.4f}]")
        print(f"Turnout Error: {abs(actual_turnout - predicted_turnout):.4f}")
        
        # Check if actual results fall within confidence intervals
        print("\nValidation Check:")
        print(f"For Share within CI: {ci['for_share'][0] <= actual_for_share <= ci['for_share'][1]}")
        print(f"Turnout within CI: {ci['turnout'][0] <= actual_turnout <= ci['turnout'][1]}")
        
        # Calculate precinct-level accuracy
        precinct_actual = self.data.groupby('PRECINCT_NAME')['nov_for_share'].mean()
        precinct_predicted = self.data.groupby('PRECINCT_NAME')['mc_mean_prediction'].mean()
        
        precinct_validation = pd.DataFrame({
            'actual': precinct_actual,
            'predicted': precinct_predicted,
            'error': abs(precinct_actual - precinct_predicted)
        })
        
        print("\nPrecinct-Level Validation:")
        print("=" * 50)
        print(f"Mean Absolute Error: {precinct_validation['error'].mean():.4f}")
        print(f"Worst Predictions:")
        print(precinct_validation.sort_values('error', ascending=False).head().to_markdown())
        
        return {
            'actual_for_share': actual_for_share,
            'predicted_for_share': predicted_for_share,
            'confidence_intervals': ci,
            'precinct_validation': precinct_validation
        }

    def analyze_current_may_turnout(self):
        """Analyze current May election turnout and predict votes"""
        # Get voters who have already voted in May
        current_voters = self.data[self.data['VOTED_MAY_LEVY'] == 1].copy()
        
        # Calculate November results by ward and precinct
        nov_ward_results = self.data.groupby('WARD').agg({
            'nov_for_share': 'mean',
            'VOTED_NOV_LEVY': 'mean'
        }).rename(columns={'nov_for_share': 'ward_nov_for_share', 'VOTED_NOV_LEVY': 'ward_nov_turnout'})
        
        nov_precinct_results = self.data.groupby('PRECINCT_NAME').agg({
            'nov_for_share': 'mean',
            'VOTED_NOV_LEVY': 'mean'
        }).rename(columns={'nov_for_share': 'precinct_nov_for_share', 'VOTED_NOV_LEVY': 'precinct_nov_turnout'})
        
        # Add November results to current voters
        current_voters = current_voters.merge(nov_ward_results, left_on='WARD', right_index=True)
        current_voters = current_voters.merge(nov_precinct_results, left_on='PRECINCT_NAME', right_index=True)
        
        # Calculate relative performance metrics (similar to decision tree)
        current_voters['precinct_vs_ward'] = (
            current_voters['precinct_nov_for_share'] - 
            current_voters['ward_nov_for_share']
        )
        
        current_voters['precinct_z_score'] = (
            current_voters['precinct_nov_for_share'] - 
            current_voters['ward_nov_for_share']
        ) / current_voters['ward_nov_for_share'].std()
        
        print("\nCurrent May Election Analysis:")
        print("=" * 50)
        print(f"Total Voters So Far: {len(current_voters)}")
        
        # Analyze turnout by demographics
        print("\nTurnout by Age Group:")
        age_turnout = current_voters.groupby('AGE_RANGE').size().to_frame('count')
        age_turnout['percentage'] = age_turnout['count'] / len(current_voters) * 100
        print(age_turnout.to_markdown())
        
        print("\nTurnout by Party:")
        party_turnout = current_voters.groupby('PARTY_CAT').size().to_frame('count')
        party_turnout['percentage'] = party_turnout['count'] / len(current_voters) * 100
        print(party_turnout.to_markdown())
        
        # Run simulations for current voters
        simulation_results = {
            'for_votes': [],
            'against_votes': [],
            'for_share': [],
            'precinct_results': {},
            'ward_results': {}
        }
        
        # Initialize precinct and ward tracking
        precincts = current_voters['PRECINCT_NAME'].unique()
        wards = current_voters['WARD'].unique()
        
        for precinct in precincts:
            simulation_results['precinct_results'][precinct] = {
                'for_votes': [],
                'against_votes': [],
                'for_share': [],
                'nov_for_share': nov_precinct_results.loc[precinct, 'precinct_nov_for_share'],
                'nov_turnout': nov_precinct_results.loc[precinct, 'precinct_nov_turnout'],
                'precinct_vs_ward': current_voters[current_voters['PRECINCT_NAME'] == precinct]['precinct_vs_ward'].mean(),
                'precinct_z_score': current_voters[current_voters['PRECINCT_NAME'] == precinct]['precinct_z_score'].mean()
            }
        
        for ward in wards:
            simulation_results['ward_results'][ward] = {
                'for_votes': [],
                'against_votes': [],
                'for_share': [],
                'nov_for_share': nov_ward_results.loc[ward, 'ward_nov_for_share'],
                'nov_turnout': nov_ward_results.loc[ward, 'ward_nov_turnout']
            }
        
        # Run simulations
        for _ in range(self.config.n_simulations):
            # Simulate votes for current voters
            simulated_votes = current_voters.apply(
                lambda row: self.simulate_sentiment(row['vote_prediction']),
                axis=1
            )
            
            # Calculate overall results
            for_votes = (simulated_votes > 0.5).sum()
            against_votes = (simulated_votes <= 0.5).sum()
            for_share = for_votes / len(simulated_votes)
            
            simulation_results['for_votes'].append(for_votes)
            simulation_results['against_votes'].append(against_votes)
            simulation_results['for_share'].append(for_share)
            
            # Calculate precinct-level results
            for precinct in precincts:
                precinct_votes = simulated_votes[current_voters['PRECINCT_NAME'] == precinct]
                precinct_for = (precinct_votes > 0.5).sum()
                precinct_against = (precinct_votes <= 0.5).sum()
                precinct_share = precinct_for / len(precinct_votes) if len(precinct_votes) > 0 else 0
                
                simulation_results['precinct_results'][precinct]['for_votes'].append(precinct_for)
                simulation_results['precinct_results'][precinct]['against_votes'].append(precinct_against)
                simulation_results['precinct_results'][precinct]['for_share'].append(precinct_share)
            
            # Calculate ward-level results
            for ward in wards:
                ward_votes = simulated_votes[current_voters['WARD'] == ward]
                ward_for = (ward_votes > 0.5).sum()
                ward_against = (ward_votes <= 0.5).sum()
                ward_share = ward_for / len(ward_votes) if len(ward_votes) > 0 else 0
                
                simulation_results['ward_results'][ward]['for_votes'].append(ward_for)
                simulation_results['ward_results'][ward]['against_votes'].append(ward_against)
                simulation_results['ward_results'][ward]['for_share'].append(ward_share)
        
        # Calculate confidence intervals
        ci_for_share = np.percentile(simulation_results['for_share'], [2.5, 97.5])
        mean_for_share = np.mean(simulation_results['for_share'])
        
        print("\nOverall Prediction:")
        print("=" * 50)
        print(f"Predicted For Share: {mean_for_share:.1%}")
        print(f"95% Confidence Interval: [{ci_for_share[0]:.1%}, {ci_for_share[1]:.1%}]")
        
        # Precinct-level predictions
        print("\nPrecinct-Level Predictions:")
        print("=" * 50)
        precinct_predictions = []
        for precinct in precincts:
            mean_share = np.mean(simulation_results['precinct_results'][precinct]['for_share'])
            ci = np.percentile(simulation_results['precinct_results'][precinct]['for_share'], [2.5, 97.5])
            count = len(current_voters[current_voters['PRECINCT_NAME'] == precinct])
            nov_share = simulation_results['precinct_results'][precinct]['nov_for_share']
            z_score = simulation_results['precinct_results'][precinct]['precinct_z_score']
            
            precinct_predictions.append({
                'precinct': precinct,
                'voters': count,
                'for_share': mean_share,
                'nov_share': nov_share,
                'change': mean_share - nov_share,
                'z_score': z_score,
                'ci_low': ci[0],
                'ci_high': ci[1]
            })
        
        precinct_df = pd.DataFrame(precinct_predictions)
        precinct_df = precinct_df.sort_values('for_share', ascending=False)
        print(precinct_df.to_markdown(floatfmt='.1%'))
        
        # Ward-level predictions
        print("\nWard-Level Predictions:")
        print("=" * 50)
        ward_predictions = []
        for ward in wards:
            mean_share = np.mean(simulation_results['ward_results'][ward]['for_share'])
            ci = np.percentile(simulation_results['ward_results'][ward]['for_share'], [2.5, 97.5])
            count = len(current_voters[current_voters['WARD'] == ward])
            nov_share = simulation_results['ward_results'][ward]['nov_for_share']
            
            ward_predictions.append({
                'ward': ward,
                'voters': count,
                'for_share': mean_share,
                'nov_share': nov_share,
                'change': mean_share - nov_share,
                'ci_low': ci[0],
                'ci_high': ci[1]
            })
        
        ward_df = pd.DataFrame(ward_predictions)
        ward_df = ward_df.sort_values('for_share', ascending=False)
        print(ward_df.to_markdown(floatfmt='.1%'))
        
        return simulation_results

    def run_simulation(self):
        """Run Monte Carlo simulation for all voters"""
        print("\nRunning Monte Carlo simulation...")
        self.simulation_results = []
        n_voters = len(self.data)
        
        # Pre-calculate base probabilities for vote predictions
        base_probs = {
            'strongly_for': 0.8,
            'swing_for': 0.65,
            'lean_for': 0.55,
            'swing': 0.5,
            'lean_against': 0.45,
            'swing_against': 0.35,
            'strongly_against': 0.2
        }
        vote_base_probs = np.array([base_probs.get(pred, 0.5) for pred in self.data['vote_prediction']])
        
        # Pre-calculate ward and precinct ratios for turnout
        ward_data = self.data.groupby('WARD').agg({
            'P_SCORE': 'mean',
            'nov_for_share': 'mean'
        })
        ward_primary_ratios = (ward_data['P_SCORE'] / ward_data['nov_for_share']).fillna(0.5)
        ward_ratios = self.data['WARD'].map(ward_primary_ratios).values
        
        # Convert P_SCORE to probability array
        p_score_probs = self.data['P_SCORE'].values / 100.0
        nov_shares = self.data['nov_for_share'].values
        
        # Use tqdm for progress tracking
        for _ in tqdm(range(self.config.n_simulations), desc="Simulating"):
            # Vectorized turnout simulation
            weighted_probs = (
                0.7 * p_score_probs +  # Primary data
                0.3 * (nov_shares * ward_ratios)  # November data adjusted by ward ratio
            )
            turnout_variation = np.random.normal(0, self.config.turnout_std, n_voters)
            turnout_probs = np.clip(weighted_probs + turnout_variation, 0, 1)
            turnout_mask = np.random.random(n_voters) < turnout_probs
            
            # Vectorized vote simulation
            vote_variation = np.random.normal(0, self.config.sentiment_std, n_voters)
            vote_probs = np.clip(vote_base_probs + vote_variation, 0, 1)
            votes = vote_probs[turnout_mask]
            
            result = {
                'turnout': turnout_mask.mean(),
                'vote_prob': (votes > 0.5).mean() if len(votes) > 0 else 0.5,
                'for_votes': (votes > 0.5).sum(),
                'against_votes': (votes <= 0.5).sum(),
                'total_votes': len(votes)
            }
            
            self.simulation_results.append(result)
        
        # Calculate and store average results
        self.avg_turnout = np.mean([r['turnout'] for r in self.simulation_results])
        self.avg_vote_prob = np.mean([r['vote_prob'] for r in self.simulation_results])
        
        print(f"\nSimulation complete.")
        print(f"Average turnout: {self.avg_turnout:.1%}")
        print(f"Average vote probability: {self.avg_vote_prob:.1%}")
        
        return self.simulation_results

    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for simulation results"""
        if not self.simulation_results:
            self.run_simulation()
            
        # Calculate confidence intervals for turnout and vote probability
        turnouts = [r['turnout'] for r in self.simulation_results]
        vote_probs = [r['vote_prob'] for r in self.simulation_results]
        
        turnout_ci = np.percentile(turnouts, [2.5, 97.5])
        vote_prob_ci = np.percentile(vote_probs, [2.5, 97.5])
        
        return {
            'turnout': (turnout_ci[0], turnout_ci[1]),
            'for_share': (vote_prob_ci[0], vote_prob_ci[1])
        }
    
    def print_summary_statistics(self):
        """Print summary statistics of the simulation results."""
        if not self.simulation_results:
            self.run_simulation()
            
        turnouts = [r['turnout'] for r in self.simulation_results]
        vote_probs = [r['vote_prob'] for r in self.simulation_results]
        
        print("\nSimulation Summary Statistics:")
        print("=" * 50)
        print(f"Average Turnout: {np.mean(turnouts):.2%}")
        print(f"Average Vote For: {np.mean(vote_probs):.2%}")
        print(f"Turnout Standard Deviation: {np.std(turnouts):.2%}")
        print(f"Vote Standard Deviation: {np.std(vote_probs):.2%}")
        
        # Calculate confidence intervals
        ci = self.get_confidence_intervals()
        print(f"\nConfidence Intervals (95%):")
        print(f"Turnout: [{ci['turnout'][0]:.2%} - {ci['turnout'][1]:.2%}]")
        print(f"Vote For: [{ci['for_share'][0]:.2%} - {ci['for_share'][1]:.2%}]")
    
    def plot_distributions(self):
        """Plot distributions of turnout and vote probabilities."""
        if not hasattr(self, 'simulation_results'):
            raise ValueError("No simulation results found. Run simulation first.")
        
        # Extract arrays from simulation results
        turnouts = [result['turnout'] for result in self.simulation_results]
        vote_probs = [result['vote_prob'] for result in self.simulation_results]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot turnout distribution
        sns.histplot(turnouts, bins=30, ax=ax1)
        ax1.set_title('Distribution of Turnout')
        ax1.set_xlabel('Turnout Rate')
        ax1.set_ylabel('Count')
        ax1.axvline(np.mean(turnouts), color='r', linestyle='--', label='Mean')
        ax1.legend()
        
        # Plot vote probability distribution
        sns.histplot(vote_probs, bins=30, ax=ax2)
        ax2.set_title('Distribution of Vote Probability')
        ax2.set_xlabel('Vote Probability')
        ax2.set_ylabel('Count')
        ax2.axvline(np.mean(vote_probs), color='r', linestyle='--', label='Mean')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_voter_predictions(self) -> pd.DataFrame:
        """Return final voter predictions based on simulation results."""
        if not self.simulation_results:
            self.run_simulation()
            
        predictions = self.data.copy()
        predictions['mc_turnout_prob'] = np.mean([r['turnout'] for r in self.simulation_results])
        predictions['mc_vote_prob'] = np.mean([r['vote_prob'] for r in self.simulation_results])
        
        # Add Monte Carlo vote decision with more nuanced thresholds
        predictions['mc_vote_decision'] = pd.cut(
            predictions['mc_vote_prob'],
            bins=[-np.inf, 0.35, 0.45, 0.48, 0.52, 0.55, 0.65, np.inf],
            labels=[
                'strongly_against',
                'lean_against',
                'swing_against',
                'swing',
                'swing_for',
                'lean_for',
                'strongly_for'
            ]
        )
        
        return predictions

    def calculate_summary_statistics(self):
        """Calculate summary statistics from simulation results."""
        # Calculate average turnout probability for each voter
        self.turnout_probabilities = np.mean(self.turnout_results, axis=1)
        
        # Calculate average vote probability for each voter (among those who turned out)
        turnout_mask = self.turnout_results > 0
        self.vote_probabilities = np.zeros(len(self.data))
        for i in range(len(self.data)):
            turnouts = turnout_mask[i]
            if np.any(turnouts):
                self.vote_probabilities[i] = np.mean(self.vote_results[i][turnouts])
        
        # Calculate overall statistics
        self.avg_turnout = np.mean(self.turnout_probabilities)
        self.avg_vote_prob = np.mean(self.vote_probabilities)
        
        # Calculate confidence intervals
        self.turnout_ci = np.percentile(np.mean(self.turnout_results, axis=0), [2.5, 97.5])
        self.vote_ci = np.percentile(np.mean(self.vote_results, axis=0), [2.5, 97.5])
        
        # Print summary
        print("\nSimulation Summary:")
        print(f"Average Turnout: {self.avg_turnout:.1%} (95% CI: {self.turnout_ci[0]:.1%} - {self.turnout_ci[1]:.1%})")
        print(f"Average Vote Probability: {self.avg_vote_prob:.1%} (95% CI: {self.vote_ci[0]:.1%} - {self.vote_ci[1]:.1%})") 