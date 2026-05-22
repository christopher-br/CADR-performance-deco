# Load modules
import os
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
os.chdir(root / "results")

path = "exp_tcga_2.csv"

data = pd.read_csv(path)

means = (
    data.groupby(["treatment_bias", "rm_confounding_t", "dose_bias", "rm_confounding_d"])
    .mean()
    .round(2)
    .sort_values("rm_confounding_d", ascending=False)
    .sort_values("dose_bias", ascending=True)
    .sort_values("rm_confounding_t", ascending=False)
    .sort_values("treatment_bias", ascending=True)
    .transpose()
    .sort_index()
)

stds = (
    data.groupby(["treatment_bias", "rm_confounding_t", "dose_bias", "rm_confounding_d"])
    .std()
    .round(2)
    .sort_values("rm_confounding_d", ascending=False)
    .sort_values("dose_bias", ascending=True)
    .sort_values("rm_confounding_t", ascending=False)
    .sort_values("treatment_bias", ascending=True)
    .transpose()
    .sort_index()
)

strs = means.astype(str) + " ± \scriptsize{" + stds.astype(str) +"}"

print(strs.to_markdown())

print(strs.to_latex())