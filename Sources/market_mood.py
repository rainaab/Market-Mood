import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# 1. load data
df = pd.read_csv("data/raw/SP500.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# 2. cleaning data

# Convert observation_date from text to datetime
df["observation_date"] = pd.to_datetime(df["observation_date"])

# Remove rows where the S&P 500 value is missing
df = df.dropna(subset=["SP500"])

# Sort observations chronologically
df = df.sort_values("observation_date")

# Reset row numbers after removing missing observations
df = df.reset_index(drop=True)

print("\nAfter cleaning:")
print(df.info())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFirst 5 cleaned rows:")
print(df.head())

print("\nLast 5 cleaned rows:")
print(df.tail())


# 3. feature engineering

# Daily percentage return
df["daily_return"] = df["SP500"].pct_change()

# 5-day rolling return (approximately one trading week)
df["return_5d"] = df["SP500"].pct_change(periods=5)

# 20-day rolling return (approximately one trading month)
df["return_20d"] = df["SP500"].pct_change(periods=20)

# 20-day rolling volatility
df["volatility_20d"] = df["daily_return"].rolling(window=20).std()

print("\nData with engineered features:")
print(df.head(25))

print("\nMissing values after feature engineering:")
print(df.isnull().sum())

# 20-day moving average
df["moving_avg_20d"] = df["SP500"].rolling(window=20).mean()

# Momentum: position relative to 20-day moving average
df["momentum_20d"] = (df["SP500"] / df["moving_avg_20d"]) - 1

# 4. prepare modeling

# Remove rows with missing engineered features
model_df = df.dropna().copy()

# Reset index
model_df = model_df.reset_index(drop=True)

print("\nFinal modeling dataset:")
print(model_df.head())

print("\nModeling dataset shape:")
print(model_df.shape)

print("\nMissing values:")
print(model_df.isnull().sum())

# Save processed dataset
model_df.to_csv("data/processed/market_mood_processed.csv", index=False)

print("\nProcessed dataset saved successfully.")

# ---------------------------------------------------------------------------------------
# 5. exploratory data analysis

print("\nDescriptive statistics:")
print(model_df.describe())

# Plot S&P 500 closing values over time
plt.figure(figsize=(12, 6))

plt.plot(
    model_df["observation_date"],
    model_df["SP500"]
)

plt.title("S&P 500 Closing Value Over Time")
plt.xlabel("Date")
plt.ylabel("S&P 500 Index")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/sp500_over_time.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Plot 20-day rolling volatility
plt.figure(figsize=(12, 6))

plt.plot(
    model_df["observation_date"],
    model_df["volatility_20d"]
)

plt.title("S&P 500 20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/rolling_volatility.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Distribution of daily returns
plt.figure(figsize=(10, 6))

plt.hist(
    model_df["daily_return"],
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of S&P 500 Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/daily_returns_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

feature_columns = [
    "daily_return",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "momentum_20d"
]

print("\nFeature correlations:")
print(model_df[feature_columns].corr().round(3))


# 6. prepare for clustering

feature_columns = [
    "daily_return",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "momentum_20d"
]

X = model_df[feature_columns]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardized feature means:")
print(X_scaled.mean(axis=0))

print("\nStandardized feature standard deviations:")
print(X_scaled.std(axis=0))


# 7. k-means

k_values = range(2, 7)

inertias = []
silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(
        f"k = {k}: "
        f"Inertia = {kmeans.inertia_:.2f}, "
        f"Silhouette Score = {score:.3f}"
    )


# 8. initial analysis

final_kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

model_df["cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nNumber of observations in each cluster:")
print(model_df["cluster"].value_counts().sort_index())

cluster_summary = model_df.groupby("cluster")[feature_columns].mean()

print("\nAverage characteristics of each cluster:")
print(cluster_summary.round(4))

cluster_summary_percent = cluster_summary.copy()

percent_columns = [
    "daily_return",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "momentum_20d"
]

cluster_summary_percent[percent_columns] *= 100

print("\nCluster characteristics (%):")
print(cluster_summary_percent.round(2))

# graph conversion
# Assign readable preliminary regime names
regime_names = {
    0: "Growth / Stable",
    1: "Decline / Stress"
}

model_df["regime"] = model_df["cluster"].map(regime_names)

# Plot market regimes over time
plt.figure(figsize=(14, 7))

for cluster in sorted(model_df["cluster"].unique()):

    cluster_data = model_df[model_df["cluster"] == cluster]

    plt.scatter(
        cluster_data["observation_date"],
        cluster_data["SP500"],
        s=12,
        label=regime_names[cluster]
    )

plt.plot(
    model_df["observation_date"],
    model_df["SP500"],
    alpha=0.25
)

plt.title("Market Mood: Discovered S&P 500 Market Regimes")
plt.xlabel("Date")
plt.ylabel("S&P 500 Index")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/market_regimes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ---------------------------------------------------------------------------------------
# 9. experiment 2, feature removal

# remove momentum to reduce feature redundancy
reduced_features = [
    "daily_return",
    "return_5d",
    "return_20d",
    "volatility_20d"
]

# standardize the reduced feature set
X_reduced = model_df[reduced_features]

reduced_scaler = StandardScaler()
X_reduced_scaled = reduced_scaler.fit_transform(X_reduced)

# evaluate K-Means using the reduced features
print("\nExperiment 2: Reduced Feature K-Means")

for k in range(2, 7):

    reduced_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    reduced_labels = reduced_kmeans.fit_predict(
        X_reduced_scaled
    )

    reduced_score = silhouette_score(
        X_reduced_scaled,
        reduced_labels
    )

    print(
        f"k = {k}: "
        f"Inertia = {reduced_kmeans.inertia_:.2f}, "
        f"Silhouette = {reduced_score:.3f}"
    )

# ---------------------------------------------------------------------------------------
# 10. experiment 3: hierarchical clustering
print("\nExperiment 3: Hierarchical Clustering")

for k in range(2, 7):

    hierarchical = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        hierarchical_labels
    )

    print(
        f"k = {k}: "
        f"Silhouette = {score:.3f}"
    )

# compare K-Means and hierarchical clustering at k=2
hierarchical_final = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

hierarchical_labels = hierarchical_final.fit_predict(
    X_scaled
)

agreement = adjusted_rand_score(
    model_df["cluster"],
    hierarchical_labels
)

print(
    "\nAdjusted Rand Index (K-Means vs Hierarchical):",
    round(agreement, 3)
)


# ---------------------------------------------------------------------------------------
# 11. hierarchical regime analysis
# store hierarchical cluster assignments
model_df["hierarchical_cluster"] = hierarchical_labels

# count observations in each cluster
print("\nHierarchical cluster counts:")
print(
    model_df["hierarchical_cluster"]
    .value_counts()
    .sort_index()
)

# calculate average characteristics
hierarchical_summary = model_df.groupby(
    "hierarchical_cluster"
)[feature_columns].mean()

# convert decimal returns and volatility to percentages
hierarchical_summary *= 100

print("\nHierarchical cluster characteristics (%):")
print(hierarchical_summary.round(2))

# compare assignments between algorithms
print("\nK-Means vs Hierarchical assignments:")
print(
    pd.crosstab(
        model_df["cluster"],
        model_df["hierarchical_cluster"]
    )
)

# ---------------------------------------------------------------------------------------
# 12. analyze cluster disagreements

# identify observations assigned differently
disagreements = model_df[
    model_df["cluster"] !=
    model_df["hierarchical_cluster"]
].copy()

print("\nNumber of disagreements:")
print(len(disagreements))

# examine the average characteristics
disagreement_summary = (
    disagreements[feature_columns].mean() * 100
)

print("\nDisagreement characteristics (%):")
print(disagreement_summary.round(2))

# examine the dates and market values
print("\nFirst 10 disagreement observations:")
print(
    disagreements[
        ["observation_date", "SP500"] +
        feature_columns
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------------------------------------
# 13. final market mood visualization

import numpy as np

# assign comparison-based market conditions
conditions = [
    (model_df["cluster"] == 0) &
    (model_df["hierarchical_cluster"] == 0),

    (model_df["cluster"] == 1) &
    (model_df["hierarchical_cluster"] == 0),

    (model_df["cluster"] == 1) &
    (model_df["hierarchical_cluster"] == 1)
]

labels = [
    "Growth / Stable",
    "Mild Decline / Transitional",
    "Pronounced Decline / Stress"
]

model_df["market_mood"] = np.select(
    conditions,
    labels,
    default="Unclassified"
)

# visualize the comparison-based categories
plt.figure(figsize=(15, 7))

colors = {
    "Growth / Stable": "green",
    "Mild Decline / Transitional": "orange",
    "Pronounced Decline / Stress": "red"
}

plt.plot(
    model_df["observation_date"],
    model_df["SP500"],
    color="gray",
    alpha=0.3
)

for mood in labels:
    subset = model_df[
        model_df["market_mood"] == mood
    ]

    plt.scatter(
        subset["observation_date"],
        subset["SP500"],
        s=14,
        color=colors[mood],
        label=mood
    )

plt.title(
    "Market Mood: Comparison-Based Market Conditions"
)
plt.xlabel("Date")
plt.ylabel("S&P 500 Index")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/final_market_moods.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
