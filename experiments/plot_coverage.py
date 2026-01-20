# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse

# %%
# --- load data ---
# fname = "../results/glm_confints_p=5_level=0.9_n=50,100,200_reps=1000_gamma=1.0_s=2.0_lam=2.0_fam=logistic.csv"
# fname = "../results/glm_confints_p=5_level=0.9_n=50,100,200_reps=100_gamma=1.0_s=2.0_lam=2.0_fam=logistic.csv"
# fname = "../results/glm_confints_p=40_level=0.95_n=50,100,200_reps=100_gamma=1.0_s=5.0_lam=5.0_fam=logistic_true_noise_var=True.csv"
# fname = "../results/glm_confints_p=40_level=0.9_n=50,100,200_reps=100_gamma=1.0_s=5.0_lam=5.0_fam=linear_errors=homogeneous_true_noise_var=False.csv"
# ref = "../results/glm_confints_p=40_level=0.9_n=50,100,200_reps=100_gamma=1.0_s=5.0_lam=5.0_fam=linear_errors=homogeneous_true_noise_var=True.csv"


def load_data(fname):
    df = pd.read_csv(fname)

    print(f"Fraction model selected: {sum(df['num_selected'] > 0) / df.shape[0]}")
    df = df[df["num_selected"] > 0]

    df["coverage"] = df["sum_covers"] / df["num_selected"]
    df["length"] = df["sum_length"] / df["num_selected"]
    # df["bias"] = df["sum_bias"] / df["num_selected"]

    coverage_df = df.groupby(["n", "method"], as_index=False).agg(
        sum_selected=("num_selected", "sum"),
        # sum_rejects=("sum_rejects", "sum"),
        sum_covers=("sum_covers", "sum"),
        # length=("length", "median"),
        # length_q05=("length", lambda x: x.quantile(0.05)),
        # length_q95=("length", lambda x: x.quantile(0.95)),
        # mean_TPR=("TPR", "mean"),
        # mean_FDR=("FDR", "mean"),
        # mean_bias=("bias", "mean"),
    )

    coverage_df["coverage"] = coverage_df["sum_covers"] / coverage_df["sum_selected"]
    # coverage_df["reject_rate"] = (
    #     coverage_df["sum_rejects"] / coverage_df["sum_selected"]
    # )
    print(coverage_df)

    return df


# %%
def main(
    fname, out_path, ref=None, level=0.9, x="n", x_label="Sample Size (n)", logx=False, exclude=None
):

    coverage_df = load_data(fname)
    coverage_df["Variance"] = "Estimated"

    if ref is not None:
        ref_df = load_data(ref)
        ref_df["Variance"] = "True"
        ref_df = ref_df[~ref_df["method"].isin(["classic", "sample_splitting"])]
        coverage_df = pd.concat([coverage_df, ref_df], axis=0)

    if exclude is not None:
        coverage_df = coverage_df[~coverage_df["method"].isin([exclude])]

    coverage_df.rename(columns={"method": "Method"}, inplace=True)
    coverage_df["Method"] = coverage_df["Method"].replace(
        {
            "classic": "Classic",
            "rsc": "Huang et al. [2024]",
            "rsc_exact": "Panigrahi et al. [2024]",  # Bakshi et al. [2024] for logistic
            "sample_splitting": "Sample Splitting",
            "thin_gradient": "Thin (grad)",
            "thin_outcomes": "Thinning (Alg 1)",
        }
    )

    method_colors = {
        "Classic": "#1f77b4",
        "Huang et al. [2024]": "#ff7f0e",
        "Sample Splitting": "#2ca02c",
        "Thin (grad)": "#d62728",
        "Thinning (Alg 1)": "#9467bd",
        "Panigrahi et al. [2024]": "#a65628",
    }

    # coverage_df = coverage_df.dropna()
    coverage_df = coverage_df[coverage_df["Method"] != "Thin (grad)"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2), width_ratios=[1, 1.5, 1])
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,  # axis label font size
            "axes.titlesize": 12,
            "xtick.labelsize": 5,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    sns.lineplot(
        data=coverage_df,
        x=x,
        y="coverage",
        hue="Method",
        marker="o",
        ax=axes[0],
        style="Variance",
        palette=method_colors,
        errorbar=None,
    )
    axes[0].axhline(level, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Coverage")
    # axes[0].set_ylim(0, 1.05)

    sns.boxplot(
        data=coverage_df[coverage_df["Method"] != "Classic"],
        x=x,
        y="length",
        hue="Method",
        # marker="o",
        ax=axes[1],
        # style="Variance",
        palette=method_colors,
        # flierprops={"marker": ".", "markersize": 5},
        showfliers=False,
    )
    axes[1].set_ylabel("CI Length")
    # Add error bars using quantiles
    # for method in coverage_df["Method"].unique():
    #     # for variance in coverage_df["Variance"].unique() if "Variance" in coverage_df.columns else [None]:
    #     #     if variance:
    #     if method in ["Classic", "Sample Splitting"]:
    #         mask = (coverage_df["Method"] == method) & (coverage_df["Variance"] == "Estimated")
    #     else:
    #         mask = (coverage_df["Method"] == method) & (coverage_df["Variance"] == "True")
    #     # else:
    #     #     mask = coverage_df["Method"] == method

    #     subset = coverage_df[mask].sort_values("n")

    #     yerr_lower = subset["length_q05"]
    #     yerr_upper = subset["length_q95"]
    #     axes[1].errorbar(subset["n"] + np.random.uniform(-5, 5, len(subset)),
    #             subset["length"],
    #             yerr=[yerr_lower, yerr_upper], color="Method",
    #             fmt="none", capsize=3, alpha=0.3)

    # sns.lineplot(
    #     data=coverage_df,
    #     x=x,
    #     y="mean_bias",
    #     hue="Method",
    #     marker="o",
    #     ax=axes[2],
    #     style="Variance",
    #     palette=method_colors,
    # )
    # axes[2].set_ylabel("Mean bias")

    # sns.lineplot(
    #     data=coverage_df, #[coverage_df["Method"] != "Classic"],
    #     x=x,
    #     y="mean_TPR",
    #     hue="Method",
    #     marker="o",
    #     ax=axes[2],
    #     style="Variance",
    #     palette=method_colors,
    # )
    # axes[2].set_ylabel("Mean TPR")

    sns.lineplot(
        data=coverage_df[coverage_df["Method"] != "Classic"],
        x=x,
        y="FDR",
        hue="Method",
        marker="o",
        ax=axes[2],
        style="Variance",
        palette=method_colors,
        errorbar=None,
    )
    axes[2].set_ylabel("Mean FDR")

    handles, labels = axes[0].get_legend_handles_labels()

    unique_methods = coverage_df["Method"].unique()
    unique_variances = coverage_df["Variance"].unique()

    num_hue = len(unique_methods) + 1
    num_style = len(unique_variances) + 1

    hue_handles = handles[:num_hue]
    hue_labels = labels[:num_hue]

    style_handles = handles[num_hue : num_hue + num_style]
    style_labels = labels[num_hue : num_hue + num_style]

    # --- First row: hue legend ---
    leg1 = fig.legend(
        hue_handles[1:],  # removes "title"
        hue_labels[1:],
        loc="lower center",
        ncol=3,#len(hue_labels),
        bbox_to_anchor=(0.5, -0.22),
        title = "",
        # frameon=False,
    )
    fig.add_artist(leg1)  # ensure legend stays on figure

    if ref is not None:
        # --- Second row: style legend ---
        leg2 = fig.legend(
            style_handles,
            style_labels,
            loc="lower center",
            ncol=len(style_labels),
            bbox_to_anchor=(0.5, -0.24),
        )
        fig.add_artist(leg2)

    # fig.legend(
    #     handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1)
    # )
    for i, ax in enumerate(axes):
        ax.legend().remove()
        ax.set_xlabel(x_label)
        if i in [2]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{np.round(x, 2)}' if x != 0 else '0'))
        # ax.tick_params(axis='x', rotation=45)  # Rotate x-tick labels
        # ax.tick_params(axis='x', labelsize=8)  # Set x-tick label font size
        if logx and i != 1:
            ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--out", type=str)
    parser.add_argument("--x", type=str, default="n")
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--x_label", type=str, default="Sample Size (n)")
    parser.add_argument("--logx", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.fname,
        args.out,
        args.reference,
        x=args.x,
        x_label=args.x_label,
        logx=args.logx,
        exclude=args.exclude,
    )

# %%
