import pandas as pd

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