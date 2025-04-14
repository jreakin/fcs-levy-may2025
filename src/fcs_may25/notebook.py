import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from fcs_may25.data_loader import ElectionReconstruction
from fcs_may25.main import (
    analyze_turnout_by_ward,
    analyze_turnout_by_category,
    analyze_election_results,
    reconstruct_individual_voting_patterns
)

app = mo.App(title="Findlay Tax Levy Analysis")

# --- Data Loading ---
@mo.cell
def load_data():
    """Load the election data using the ElectionReconstruction class."""
    election_reconstruction = ElectionReconstruction()
    election_results = election_reconstruction.data.election_results
    november_turnout = election_reconstruction.data.turnout_data
    return election_results, november_turnout

# --- Turnout Analysis ---
@mo.cell
def analyze_turnout():
    """Analyze voter turnout by ward and category."""
    election_results, november_turnout = load_data()
    
    # Analyze turnout by ward
    ward_turnout = analyze_turnout_by_ward(november_turnout)
    
    # Analyze turnout by category
    category_turnout = analyze_turnout_by_category(november_turnout)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ward turnout visualization
    ward_turnout.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Voter Turnout by Ward')
    axes[0].set_xlabel('Ward')
    axes[0].set_ylabel('Number of Voters')
    
    # Category turnout visualization
    category_turnout.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Voter Turnout by Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Number of Voters')
    
    plt.tight_layout()
    return ward_turnout, category_turnout, fig

# --- Election Results Analysis ---
@mo.cell
def analyze_results():
    """Analyze election results by ward and category."""
    election_results, november_turnout = load_data()
    
    # Analyze election results
    ward_results, category_results = analyze_election_results(election_results, november_turnout)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ward results visualization
    ward_results.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Election Results by Ward')
    axes[0].set_xlabel('Ward')
    axes[0].set_ylabel('Percentage')
    
    # Category results visualization
    category_results.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Election Results by Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Percentage')
    
    plt.tight_layout()
    return ward_results, category_results, fig

# --- Individual Voting Pattern Reconstruction ---
@mo.cell
def reconstruct_patterns():
    """Reconstruct individual voting patterns based on ward and category analysis."""
    election_results, november_turnout = load_data()
    
    # Reconstruct individual voting patterns
    voter_predictions, category_summary, ward_summary = reconstruct_individual_voting_patterns(
        election_results, november_turnout
    )
    
    # Display summary statistics
    print("Category Summary:")
    print(category_summary)
    print("\nWard Summary:")
    print(ward_summary)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Category summary visualization
    category_summary.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Voting Patterns by Category')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Percentage')
    
    # Ward summary visualization
    ward_summary.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Voting Patterns by Ward')
    axes[1].set_xlabel('Ward')
    axes[1].set_ylabel('Percentage')
    
    plt.tight_layout()
    return voter_predictions, category_summary, ward_summary, fig

# --- Run the notebook ---
if __name__ == "__main__":
    app.run()
