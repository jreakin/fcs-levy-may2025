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


@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    turnout_std: float = 0.1  # Standard deviation for turnout variation
    sentiment_std: float = 0.15  # Standard deviation for sentiment variation

class MonteCarloVoterSimulation:
    def __init__(self, data: pd.DataFrame, config: MonteCarloConfig = MonteCarloConfig()):
        self.data = data
        self.config = config
        self.results = []
        self.voter_predictions = []  # Store individual voter predictions
        np.random.seed(config.random_seed)
        
    def simulate_turnout(self, base_turnout: float) -> float:
        """Simulate turnout with random variation"""
        return np.random.normal(base_turnout, self.config.turnout_std)
    
    def simulate_sentiment(self, base_sentiment: float) -> float:
        """Simulate voter sentiment with random variation"""
        return np.random.normal(base_sentiment, self.config.sentiment_std)
    
    def run_simulation(self) -> Dict[str, List[float]]:
        """Run Monte Carlo simulation for voting predictions"""
        simulation_results = {
            'turnout': [],
            'for_votes': [],
            'against_votes': [],
            'for_share': [],
            'against_share': []
        }
        
        # Initialize voter predictions DataFrame
        self.voter_predictions = pd.DataFrame(index=self.data.index)
        
        for sim_num in range(self.config.n_simulations):
            # Simulate turnout
            simulated_turnout = self.simulate_turnout(self.data['VOTED_MAY_LEVY'].mean())
            
            # Simulate sentiment for each voter
            simulated_sentiment = self.data['vote_prediction'].apply(
                lambda x: self.simulate_sentiment(1.0 if x in ['strongly_for', 'lean_for'] else 0.0)
            )
            
            # Store individual voter predictions
            self.voter_predictions[f'sim_{sim_num}'] = simulated_sentiment
            
            # Calculate votes
            for_votes = (simulated_sentiment > 0.5).sum()
            against_votes = (simulated_sentiment <= 0.5).sum()
            total_votes = for_votes + against_votes
            
            # Store results
            simulation_results['turnout'].append(simulated_turnout)
            simulation_results['for_votes'].append(for_votes)
            simulation_results['against_votes'].append(against_votes)
            simulation_results['for_share'].append(for_votes / total_votes if total_votes > 0 else 0)
            simulation_results['against_share'].append(against_votes / total_votes if total_votes > 0 else 0)
        
        self.results = simulation_results
        return simulation_results
    
    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for simulation results"""
        if not self.results:
            self.run_simulation()
            
        confidence_intervals = {}
        for key, values in self.results.items():
            mean = np.mean(values)
            std = np.std(values)
            z_score = 1.96  # For 95% confidence interval
            margin_of_error = z_score * (std / np.sqrt(len(values)))
            confidence_intervals[key] = (mean - margin_of_error, mean + margin_of_error)
        
        return confidence_intervals
    
    def plot_distributions(self):
        """Plot distributions of simulation results"""
        if not self.results:
            self.run_simulation()
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot turnout distribution
        sns.histplot(self.results['turnout'], ax=axes[0, 0])
        axes[0, 0].set_title('Turnout Distribution')
        axes[0, 0].set_xlabel('Turnout Rate')
        
        # Plot for votes distribution
        sns.histplot(self.results['for_votes'], ax=axes[0, 1])
        axes[0, 1].set_title('For Votes Distribution')
        axes[0, 1].set_xlabel('Number of For Votes')
        
        # Plot against votes distribution
        sns.histplot(self.results['against_votes'], ax=axes[1, 0])
        axes[1, 0].set_title('Against Votes Distribution')
        axes[1, 0].set_xlabel('Number of Against Votes')
        
        # Plot for share distribution
        sns.histplot(self.results['for_share'], ax=axes[1, 1])
        axes[1, 1].set_title('For Share Distribution')
        axes[1, 1].set_xlabel('For Vote Share')
        
        plt.tight_layout()
        plt.savefig(FilePaths.IMAGE_PATH / 'monte_carlo_distributions.png')
        plt.show()
    
    def print_summary_statistics(self):
        """Print summary statistics of simulation results"""
        if not self.results:
            self.run_simulation()
            
        confidence_intervals = self.get_confidence_intervals()
        
        print("\nMonte Carlo Simulation Results:")
        print("=" * 50)
        for key, (lower, upper) in confidence_intervals.items():
            mean = np.mean(self.results[key])
            print(f"{key.replace('_', ' ').title()}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
            print(f"  Standard Deviation: {np.std(self.results[key]):.4f}")
            print("-" * 50)
    
    def get_voter_predictions(self) -> pd.DataFrame:
        """Return DataFrame with individual voter predictions"""
        if self.voter_predictions.empty:
            self.run_simulation()
        
        # Calculate mean prediction for each voter
        self.voter_predictions['mean_prediction'] = self.voter_predictions.mean(axis=1)
        
        # Calculate prediction confidence (standard deviation)
        self.voter_predictions['prediction_std'] = self.voter_predictions.std(axis=1)
        
        # Merge with original data
        result_df = self.data.copy()
        result_df['mc_mean_prediction'] = self.voter_predictions['mean_prediction']
        result_df['mc_prediction_std'] = self.voter_predictions['prediction_std']
        
        return result_df 