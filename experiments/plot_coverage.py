
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from m_estimation_SI.plotting import coverage_plots

# %%
# --- load data ---
#fname = "../results/glm_confints_p=5_level=0.9_n=50,100,200_reps=1000_gamma=1.0_s=2.0_lam=2.0_fam=logistic.csv"
# fname = "../results/glm_confints_p=5_level=0.9_n=50,100,200_reps=100_gamma=1.0_s=2.0_lam=2.0_fam=logistic.csv"
# fname = "../results/glm_confints_p=40_level=0.95_n=50,100,200_reps=100_gamma=1.0_s=5.0_lam=5.0_fam=logistic_true_noise_var=True.csv"
fname = "../results/glm_confints_p=40_level=0.9_n=50,100,200_reps=100_gamma=1.0_s=10.0_lam=5.0_fam=linear_true_noise_var=False.csv"
df = pd.read_csv(fname)

########################
# Unconditional coverage
########################

# unconditional coverage
level = 0.9

df["coverage"] = df["sum_covers"] / df["num_selected"]
df["length"] = df["sum_length"] / df["num_selected"]
df["bias"] = df["sum_bias"] / df["num_selected"]

coverage_df = (
    df.groupby(["n", "method"], as_index=False)
      .agg(
            sum_selected=("num_selected", "sum"),
            sum_rejects=("sum_rejects", "sum"),
            sum_covers=("sum_covers", "sum"),
            length=("length", "median"),
            mean_TPR=("TPR", "mean"),
            mean_FDR=("FDR", "mean"),
            med_bias=("bias", "mean"),
           )
)
coverage_df["coverage"] = coverage_df["sum_covers"] / coverage_df["sum_selected"]
coverage_df["reject_rate"] = coverage_df["sum_rejects"] / coverage_df["sum_selected"]


#%%
plt.figure(figsize=(7, 5))
    
sns.lineplot(data=coverage_df, x="n", y="coverage", hue="method", marker="o")
plt.xlabel("Sample size n")

plt.axhline(level, color="red", linestyle="--", linewidth=1)
plt.ylabel("Coverage")
plt.ylim(0, 1.05)
plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.show()

plt.figure(figsize=(7, 5))
sns.lineplot(data=coverage_df, x="n", y="length", hue="method", marker="o")
plt.xlabel("Sample size n")

plt.ylabel("Average CI Length")
plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.show()

# TPR and FDR compared to classic

plt.figure(figsize=(7, 5))

sns.lineplot(data=coverage_df, x="n", y="mean_TPR", hue="method", marker="o", linestyle="-")

plt.xlabel("Sample size n")
plt.ylabel("mean TPR")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()


plt.figure(figsize=(7, 5))

sns.lineplot(data=coverage_df, x="n", y="mean_FDR", hue="method", marker="o", linestyle="-")

plt.xlabel("Sample size n")
plt.ylabel("mean FDR")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()



# %%
