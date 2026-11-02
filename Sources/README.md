# Market Mood — Source Code & Implementation

## Project Overview

**Market Mood: Discovering Stock Market Regimes Using Unsupervised Data Mining**

Market Mood is a Python-based data mining project that investigates historical S&P 500 market behavior using unsupervised machine learning.

The project applies K-Means and hierarchical clustering to identify patterns in financial market returns, volatility, and momentum. It also compares the clustering algorithms to investigate differences in how market conditions are classified.

This directory contains the source code, datasets, and generated outputs used in the analysis.

## 1. System Requirements

### Development Environment

| Component | Requirement |
|---|---|
| Operating System | macOS (development environment); other systems may work with compatible Python installations |
| Programming Language | Python 3.9 or compatible version |
| IDE | PyCharm (used during development; optional) |
| Version Control | Git |
| Package Manager | pip |

### Python Libraries and Versions

The project was developed and tested using Python 3.9.6
with the following dependencies:

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.2.3 | Data preprocessing and analysis |
| NumPy | 2.0.2 | Numerical calculations |
| Matplotlib | 3.9.3 | Data visualization |
| scikit-learn | 1.6.1 | Clustering and evaluation |
| SciPy | 1.13.1 | Scientific computing |

The required packages are listed in the root-level `requirements.txt` file.

To display the installed library versions, run:

```bash
python3 -m pip show pandas numpy matplotlib scikit-learn scipy
```

To display the Python version:

```bash
python3 --version
```

**Version note:** The project was developed using a Python 3.9 interpreter. Exact dependency versions should be verified from the development environment and recorded in `requirements.txt` before final submission.

## 2. Directory Structure

```text
Market-Mood/
├── Reports/
│   └── README.md
├── Sources/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── outputs/
│   │   ├── figures/
│   │   └── README.md
│   ├── market_mood.py
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Descriptions

| Directory/File | Purpose |
|---|---|
| `market_mood.py` | Main Python script containing the data mining implementation |
| `data/raw/` | Original historical S&P 500 dataset |
| `data/processed/` | Cleaned and feature-engineered dataset |
| `outputs/` | Generated results and visualizations |
| `outputs/figures/` | Saved graphs illustrating market behavior and clustering results |
| `requirements.txt` | Required Python dependencies |
| `Reports/` | Academic project documentation |

## 3. Dataset

**Data Source:** Federal Reserve Economic Data (FRED)

**Dataset:** S&P 500 Index (`SP500`)

**Observation Period:** October 2021 – October 2026

The original dataset contained 1,305 observations. During preprocessing, 50 observations with missing S&P 500 values were removed, leaving 1,255 valid observations.

After feature engineering and removal of incomplete rolling-window observations, the final modeling dataset contained **1,235 observations**.

### Engineered Features

| Feature | Description |
|---|---|
| `daily_return` | Daily percentage change in the S&P 500 |
| `return_5d` | Five-trading-day percentage return |
| `return_20d` | Twenty-trading-day percentage return |
| `volatility_20d` | Twenty-day rolling standard deviation of daily returns |
| `momentum_20d` | Relative difference between the index and its twenty-day moving average |

These five behavioral features are standardized before clustering to prevent differences in numerical scale from disproportionately influencing the results.

### Data Preparation

The preprocessing pipeline performs the following operations:

1. Loads the historical S&P 500 dataset.
2. Converts observation dates to datetime format.
3. Removes observations with missing closing values.
4. Sorts observations chronologically.
5. Calculates daily returns and rolling-window indicators.
6. Removes incomplete feature observations.
7. Prepares the resulting dataset for clustering.

The original dataset should be placed in `data/raw/`, or the location specified in the script.

The processed dataset is stored in `data/processed/` when exported by the implementation.

## 4. Installation Instructions

### Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/rainaab/Market-Mood.git
```

Navigate into the project directory:

```bash
cd Market-Mood
```

### Step 2 — Create a Virtual Environment

A virtual environment is recommended to keep the project's dependencies separate from other Python installations.

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

From the root project directory:

```bash
python -m pip install -r requirements.txt
```

### Step 4 — Verify Installation

Check the Python interpreter:

```bash
python --version
```

Verify the required libraries:

```bash
python -c "import pandas, numpy, matplotlib, sklearn, scipy; print('Dependencies installed successfully')"
```

### Step 5 — Confirm Dataset Availability

Ensure the historical S&P 500 dataset is available in the expected input directory.

The dataset must contain the following columns:

- `observation_date`
- `SP500`

If the input CSV has a different filename or location, update the corresponding file path in `market_mood.py`.

## 5. Running the Project

Navigate to the `Sources` directory:

```bash
cd Sources
```

Execute the main Python script:

```bash
python market_mood.py
```

The script runs the data preprocessing, feature engineering, exploratory analysis, clustering experiments, and market-regime visualizations.

### Expected Execution Workflow

**Stage 1 — Data Loading and Cleaning**

Loads historical S&P 500 observations, inspects missing values, and prepares the dataset.

**Stage 2 — Feature Engineering**

Calculates daily returns, rolling returns, volatility, moving averages, and momentum.

**Stage 3 — Exploratory Data Analysis**

Generates descriptive statistics, correlation analysis, and historical market visualizations.

**Stage 4 — Baseline K-Means Clustering**

Standardizes the features and evaluates K-Means with cluster counts from 2 through 6.

**Stage 5 — Reduced-Feature Experiment**

Removes the 20-day momentum feature and repeats the K-Means evaluation to investigate the effect of correlated features.

**Stage 6 — Hierarchical Clustering**

Applies agglomerative hierarchical clustering and compares its silhouette scores with the K-Means results.

**Stage 7 — Cross-Model Evaluation**

Uses the Adjusted Rand Index to compare clustering assignments and identifies observations where the algorithms disagree.

**Stage 8 — Final Market Mood Visualization**

Creates a comparison-based visualization showing growth/stable, mild-decline, and pronounced-stress market conditions.

## 6. Experimental Results

The project evaluates clustering quality using silhouette scores and compares model assignments using the Adjusted Rand Index.

### Silhouette Score Comparison

| Clusters (k) | Original K-Means | Reduced K-Means | Hierarchical |
|---|---:|---:|---:|
| 2 | 0.369 | 0.332 | **0.380** |
| 3 | 0.313 | 0.330 | 0.185 |
| 4 | 0.244 | 0.236 | 0.176 |
| 5 | 0.256 | 0.237 | 0.187 |
| 6 | 0.257 | 0.242 | 0.153 |

The highest silhouette score among the tested configurations was achieved by hierarchical clustering with two clusters.

### Cross-Model Comparison

The K-Means and hierarchical clustering models produced an **Adjusted Rand Index of 0.584**, indicating partial agreement between their assignments.

The comparison identified 130 observations assigned to different market-condition groups.

### Comparison-Based Market Conditions

| Market Condition | Observations | Percentage |
|---|---:|---:|
| Growth / Stable | 900 | 72.9% |
| Mild Decline / Transitional | 130 | 10.5% |
| Pronounced Decline / Stress | 205 | 16.6% |

These categories were constructed by comparing two binary clustering models rather than by training a standalone three-cluster model.

The disagreement analysis suggests that the algorithms differ primarily in how they classify milder negative-market conditions.

## 7. Generated Outputs

The analysis produces visualizations illustrating:

- Historical S&P 500 closing values
- Distribution of daily market returns
- Twenty-day rolling volatility
- Initial K-Means market regimes
- Final comparison-based market conditions

Generated figures are saved in the `outputs/figures/` directory.

The final visualization is:

`outputs/figures/final_market_moods.png`

The visualization uses three descriptive categories to distinguish different market conditions across the historical S&P 500 timeline.

## 8. Troubleshooting

### Missing Python Packages

If a required package is unavailable, reinstall the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Dataset Not Found

Confirm that the input CSV exists and that the path in `market_mood.py` matches its location.

Relative paths may depend on the current working directory. The documented execution instructions assume the script is run from `Sources/`.

### Output Directory Not Found

Verify that the output folders exist before saving generated figures.

For example:

```python
from pathlib import Path

Path("outputs/figures").mkdir(
    parents=True,
    exist_ok=True
)
```

### Unexpected Clustering Results

Confirm that:

- The same dataset and observation period are used.
- Missing values are handled consistently.
- The same features are selected.
- Standardization is applied before clustering.
- Algorithm parameters and random seeds match the implementation.

## 9. Limitations

Market Mood is an exploratory, unsupervised data mining project.

The resulting market-condition labels are interpretations of historical clustering patterns, not verified financial classifications or predictions of future market movements.

Clustering outcomes may vary with feature selection, preprocessing decisions, model parameters, and the historical observation period.

The analysis uses S&P 500 index behavior rather than individual stock prices or external sentiment indicators.

## 10. Project Documentation

Additional project documentation is available in the `Reports/` directory, including the project proposal, mid-project report, and final project report when completed.

For the complete project overview, visit the [main README](../README.md).

## Author

**Raina Bethea**  
Georgia State University  
CSC 4740 — Data Mining  
Fall 2026
