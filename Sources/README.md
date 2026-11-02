# Market Mood — Source Code

This directory contains the Python implementation, datasets, and generated outputs for the Market Mood project.

## Main Script

**`market_mood.py`**

The main analysis script performs the following operations:

1. Loads and cleans historical S&P 500 data.
2. Engineers return, volatility, moving-average, and momentum features.
3. Performs exploratory data analysis and visualization.
4. Standardizes features for clustering.
5. Evaluates K-Means models with different cluster counts.
6. Tests a reduced feature set.
7. Evaluates hierarchical clustering.
8. Compares model assignments using the Adjusted Rand Index.
9. Analyzes disagreements between clustering algorithms.
10. Visualizes the discovered market conditions.

## Directory Structure

```text
Sources/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── market_mood.py
└── README.md
```

## Data

- **`data/raw/`** — Original historical S&P 500 data.
- **`data/processed/`** — Cleaned data and engineered features used for modeling.
- **`outputs/`** — Generated visualizations and analysis results.

## Running the Analysis

Install the dependencies listed in the project's root `requirements.txt`.

From the `Sources` directory, run:

```bash
python3 market_mood.py
```

Ensure the required input dataset is available at the path specified in the script.

For the full project overview, see the [main README](../README.md).