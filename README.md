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
  - `prediction.py`: Implements the main prediction model
    - Linear regression model for vote prediction
    - Feature engineering and selection
    - Vote share calculations by age, ward, and precinct
    - Visualization of predictions
  - `decision_tree.py`: Decision tree analysis
    - Creates interpretable decision trees
    - Analyzes voter decision-making patterns
    - Visualizes decision paths
    - Provides feature importance analysis
  - `monte_carlo.py`: Monte Carlo simulation
    - Simulates voter turnout and sentiment
    - Provides confidence intervals for predictions
    - Tracks individual voter predictions
    - Generates distribution plots
  - `main.py`: Orchestrates the analysis
    - Runs all models
    - Generates predictions and visualizations
    - Exports results to CSV files
  - `notebook.py`: Interactive Marimo notebook
    - Interactive data exploration
    - Real-time visualization
    - Custom analysis capabilities

## Analysis Components

### 1. Data Processing (`data_loader.py`)
- Loads and processes voter registration data
- Merges election results with voter records
- Creates categorical features (age ranges, party affiliation)
- Handles missing data and data cleaning
- Calculates voting history scores

### 2. Linear Prediction Model (`prediction.py`)
- Implements logistic regression for vote prediction
- Features include:
  - Age ranges
  - Party affiliation
  - Voting history
  - Ward and precinct information
- Generates predictions by:
  - Age group
  - Ward
  - Precinct
- Visualizes predictions using:
  - Vote share plots
  - Pie charts
  - Bar charts

### 3. Decision Tree Analysis (`decision_tree.py`)
- Creates interpretable decision trees
- Analyzes voter decision-making patterns
- Features:
  - Age range
  - Party affiliation
  - Ward
  - Voting history scores
- Provides:
  - Decision path analysis
  - Feature importance
  - Visual tree representation

### 4. Monte Carlo Simulation (`monte_carlo.py`)
- Simulates election outcomes with uncertainty
- Features:
  - Turnout variation
  - Sentiment variation
  - Individual voter predictions
- Outputs:
  - Confidence intervals
  - Distribution plots
  - Summary statistics
  - Individual voter predictions

### 5. Visualization and Export
- Generates plots for:
  - Vote shares by age and ward
  - Prediction distributions
  - Decision tree visualizations
  - Monte Carlo simulation results
- Exports data to CSV files:
  - Predictions by ward
  - Predictions by age
  - Predictions by precinct

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