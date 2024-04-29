# Load modules
import os
import pandas as pd

os.chdir(".../res")

path = "res:exp_ihdp_1.csv"

data = pd.read_csv(path)

means = (
    data.groupby(["bias", "rm_confounding"])
    .mean()
    .round(2)
    .sort_values("rm_confounding", ascending=False)
    .sort_values("bias", ascending=True)
    .transpose()
    .sort_index()
)

stds = (
    data.groupby(["bias", "rm_confounding"])
    .std()
    .round(2)
    .sort_values("rm_confounding", ascending=False)
    .sort_values("bias", ascending=True)
    .transpose()
    .sort_index()
)

strs = means.astype(str) + " ± " + stds.astype(str)

print(strs.to_markdown())

strs = means.astype(str) + " ± \scriptsize{" + stds.astype(str) +"}"

print(strs.to_latex())