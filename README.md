# Findlay City Schools May 2025 Tax Levy Analysis

This project analyzes voter turnout and election results for the Findlay City Schools May 2025 Tax Levy.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fcs-may25.git
cd fcs-may25
```

2. Install dependencies:
```bash
pip install -e .
```

## Running the Marimo Notebook

To run the Marimo notebook locally:

```bash
marimo edit src/fcs_may25/notebook.py
```

To export the notebook as HTML:

```bash
marimo export html-wasm src/fcs_may25/notebook.py -o src/fcs_may25/output --mode run
```

## Project Structure

- `src/fcs_may25/`: Main package directory
  - `data_loader.py`: Code for loading and processing election data
  - `main.py`: Main analysis code
  - `notebook.py`: Marimo notebook for interactive analysis

## Analysis

The analysis includes:

1. Voter turnout analysis by ward and category
2. Election results analysis by ward and category
3. Individual voting pattern reconstruction based on ward and category analysis

## GitHub Pages

The analysis is automatically deployed to GitHub Pages when changes are pushed to the main branch.
