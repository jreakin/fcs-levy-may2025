#!/bin/bash

# Check if marimo is installed
if ! command -v marimo &> /dev/null; then
    echo "Marimo is not installed. Installing..."
    pip install marimo
fi

# Create output directory if it doesn't exist
mkdir -p src/fcs_may25/output

# Run the notebook
echo "Running Marimo notebook..."
marimo edit src/fcs_may25/notebook.py

# Export the notebook as HTML
echo "Exporting notebook as HTML..."
marimo export html-wasm src/fcs_may25/notebook.py -o src/fcs_may25/output --mode run

echo "Done! The HTML export is available in src/fcs_may25/output/" 