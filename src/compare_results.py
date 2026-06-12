# src/compare_results.py

import glob
import pandas as pd


result_files = glob.glob("data/processed/model_results/*.csv")

results = []

for file in result_files:
    df = pd.read_csv(file)
    results.append(df)

comparison_df = pd.concat(results, ignore_index=True)

comparison_df = comparison_df.sort_values(
    by=["MAE"],
    ascending=True
)

print(comparison_df)

comparison_df.to_csv("data/processed/model_comparison_results.csv", index=False)