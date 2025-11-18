
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from m_estimation_SI.plotting import coverage_plots

# %%
# --- load data ---
fname = "../results/glm_confints_p=40_level=0.95_n=50,100,200_reps=100_gamma=1.0_s=5.0_lam=5.0_fam=logistic_true_noise_var=True.csv"
df = pd.read_csv(fname)

########################
# Conditional coverage
########################

#%%
# --- compute coverage ---
df["coverage"] = df["sum_covers"] / df["num_selected"]
coverage_df = (
    df.groupby(["n", "model", "method"], as_index=False)
      .agg(sum_covers=("sum_covers", "sum"),
           num_selected=("num_selected", "sum"))
)
coverage_df["coverage"] = coverage_df["sum_covers"] / coverage_df["num_selected"]

#%%
# --- plot coverage for model "1,2,4" ---
model = "0,2,3"
plt.figure()
for method, sub in coverage_df[coverage_df["model"] == model].groupby("method"):
    plt.plot(sub["n"], sub["coverage"], marker="o", label=method)
plt.xlabel("n")
plt.ylabel("Coverage")
plt.title('Coverage for model "1,2,4"')
plt.legend()
plt.show()

#%%
# --- top 5 models per n, per method ---
methods = df["method"].unique()
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, method in zip(axes, methods):
    sub = df[df["method"] == method]
    for n_val, n_group in sub.groupby("n"):
        # pick top 5 models by frequency of selection
        top_models = (
            n_group.groupby("model")["num_selected"].sum()
                   .nlargest(5).index
        )
        for model in top_models:
            model_data = n_group[n_group["model"] == model]
            ax.plot(model_data["n"], model_data["num_selected"], marker="o", label=model)
    ax.set_title(method)
    ax.set_xlabel("n")
    ax.set_ylabel("Num selected")
    ax.legend(fontsize=6)

plt.tight_layout()
plt.show()

# %%
methods = df["n"].unique()
top_per_method = {}

for m in methods:
    top_per_method[m] = (
        df[df["n"] == m]
        .groupby("model")["num_selected"].sum()
        .sort_values(ascending=False)
        .head(10)
    )

# intersection across methods
common_top = set.intersection(*(set(s.index) for s in top_per_method.values()))
print("Common top models:", common_top)
print(model_counts.head(10))  # top 10 most commonly selected models

# %%

