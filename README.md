# Findlay City Schools May 2025 Tax Levy Analysis

This project provides a comprehensive analysis of voter behavior and election predictions for the Findlay City Schools May 2025 Tax Levy. The package includes multiple models and analysis tools to understand voter patterns and predict election outcomes.

## Setup

1. Install UV (if not already installed):
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/jreakin/fcs-may25.git
cd fcs-may25
```

3. Install dependencies using UV:
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package and its dependencies
uv pip install -e .
```

## Project Structure

- `src/fcs_may25/`: Main package directory
  - `data_loader.py`: Handles data loading and preprocessing
    - Loads voter file data
    - Processes election results
    - Creates age ranges and categories
    - Handles missing data
    - Merges election results with voter data
  - `prediction.py`: Linear prediction model
  - `decision_tree_.py`: Decision tree analysis
  - `monte_carlo.py`: Monte Carlo simulation
  - `main.py`: Orchestrates the analysis
  - `notebook.py`: Interactive Marimo notebook

## Analysis Components

### 1. Linear Prediction Model (`prediction.py`)
The linear model provides a detailed analysis of individual voter behavior and historical patterns:

Key Features:
- Individual voter attributes (age, party, voting history)
- Ward and precinct-level November election results
- Interaction features (age-party, age-ward combinations)
- Weighted predictions combining model (30%) and historical results (70%)

Strengths:
- Precise individual voter predictions
- Probabilistic estimates of voting behavior
- Strong handling of continuous variables
- Direct incorporation of November results

Output Categories:
- 7-level prediction system: strongly_for, lean_for, swing_for, swing, swing_against, lean_against, strongly_against
- Thresholds adjusted based on precinct-level November performance
- Weighted combination of model predictions and historical results

### 2. Decision Tree Analysis (`decision_tree_.py`)
The decision tree model focuses on understanding precinct and ward-level voting patterns:

Key Features:
- Precinct vs. ward November performance (z-scores)
- Relative voting shares within wards
- Individual characteristics as secondary factors
- Precinct-level performance categorization

Strengths:
- Identifies non-linear relationships
- Natural handling of categorical variables
- Clear decision paths for analysis
- Strong precinct-level insights

Categories Based On:
- Absolute November Performance:
  - Strong: >60% or <40%
  - Moderate: 40-60%
- Relative to Ward (z-scores):
  - Significantly different: |z| > 1
  - Moderately different: 0.5 < |z| < 1
  - Similar: |z| < 0.5

### 3. Monte Carlo Simulation (`monte_carlo.py`)
The Monte Carlo model provides uncertainty analysis and scenario testing:

Key Features:
- Multiple election scenario simulations
- Turnout variation modeling
- Sentiment uncertainty analysis
- November results integration
- Confidence interval generation

Simulation Parameters:
- Turnout Variation: σ = 0.1 (10% standard deviation)
- Sentiment Variation: σ = 0.15 (15% standard deviation)
- 1000 simulations per run
- 95% confidence intervals

Output:
- Probability distributions for:
  - Overall turnout
  - For/Against vote totals
  - Vote share ranges
- Individual voter prediction confidence
- Precinct-level uncertainty estimates
- November vs. current comparison

### Model Comparison and Usage

Each model serves a distinct analytical purpose and provides unique insights:

1. Linear Model (`prediction.py`):
   - Best for: Individual voter predictions and probabilities
   - Use when: Need precise voter-level estimates
   - Strength: Direct incorporation of November results
   - Unique Value: Most accurate individual predictions

2. Decision Tree (`decision_tree_.py`):
   - Best for: Understanding precinct patterns and ward relationships
   - Use when: Analyzing voting segments and trends
   - Strength: Clear interpretation of voting patterns
   - Unique Value: Best for precinct-level analysis

3. Monte Carlo (`monte_carlo.py`):
   - Best for: Understanding uncertainty and risk
   - Use when: Need confidence intervals and scenario analysis
   - Strength: Comprehensive uncertainty modeling
   - Unique Value: Only model providing probability distributions

The models are complementary rather than redundant:
- Linear Model: Individual voter focus
- Decision Tree: Precinct and ward patterns
- Monte Carlo: Uncertainty and risk analysis

## Running the Analysis

### Command Line
```bash
python src/fcs_may25/main.py
```

### Interactive Notebook
```bash
marimo edit src/fcs_may25/notebook.py
```

### Exporting Results
```bash
marimo export html-wasm src/fcs_may25/notebook.py -o src/fcs_may25/output --mode run
```

## Output Files
- `predictions/`: Directory containing CSV files with predictions
- `images/`: Directory containing visualization plots
- `output/`: Directory containing exported notebook results

## GitHub Pages
The analysis is automatically deployed to GitHub Pages when changes are pushed to the main branch.