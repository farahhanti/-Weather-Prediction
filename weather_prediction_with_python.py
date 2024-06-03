import pandas as pd

weather = pd.read_csv("weather.xlsx - 3705881.csv", index_col="DATE")

weather

null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]

null_pct

weather.apply(pd.isnull).sum()

valid_columns = weather.columns[null_pct < .05]

valid_columns

weather = weather[valid_columns].copy()

weather.columns  = weather.columns.str.lower()

weather

weather = weather.ffill()

weather.apply(pd.isnull).sum()

weather.dtypes

weather.index

weather.index = pd.to_datetime(weather.index)

weather.index

weather.index.year.value_counts()

print(weather.columns)

weather["tavg"].plot()

weather

target = 'tavg'
weather['next_tavg'] = weather[target].shift(-1)

weather['next_tavg'] = weather['tavg'].shift(-1)

weather

weather = weather.ffill()

weather

from sklearn.linear_model import Ridge

rr = Ridge(alpha=.1)

predictors = weather.columns[~weather.columns.isin(["tavg", "name", "station"])]

predictors

import numpy as np

# Select only the numeric columns (excluding string columns)
numeric_cols = weather.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

print(correlation_matrix)

def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i]
        test = weather.iloc[i:(i + step)]

        model.fit(train[predictors], train["tavg"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["tavg"], preds], axis=1)

        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)

predictions = backtest(weather, rr, predictors)

predictions

from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["prediction"])

predictions["diff"].mean()

def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["tavg"]:
        weather = compute_rolling(weather, horizon, col)

weather

weather = weather.iloc[14:,:]

weather

weather = weather.fillna(0)

# Trim all column names in the DataFrame to remove leading/trailing whitespace
weather.columns = weather.columns.str.strip()

# Now your loop without hidden characters in column names
for col in ["tavg"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

weather

predictors = weather.columns[~weather.columns.isin(["tavg"])]

predictors

# Ensure that 'predictors' includes only numeric columns
numeric_cols = weather.select_dtypes(include=[np.number]).columns
predictors = [col for col in predictors if col in numeric_cols]

# Now run your backtest function with the filtered predictors
predictions = backtest(weather, rr, predictors)

mean_absolute_error(predictions["actual"], predictions["prediction"])

predictions.sort_values("diff", ascending=False)

weather.loc ["2013-08-11" : "2015-07-08" ]

predictions["diff"].round().value_counts().sort_index()

predictions["diff"].round().value_counts().sort_index().plot()
