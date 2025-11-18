import matplotlib.pyplot as plt
import seaborn as sns

def plot_selected_coeffs(beta_sel, conf_int, sel, p, title=""):
    """
    Plot point estimates and confidence intervals for selected coefficients.

    Parameters
    ----------
    beta_sel : array, shape (k,)
        Estimated coefficients for selected features.
    conf_int : array, shape (k, 2)
        Confidence intervals (lower, upper).
    sel : array, shape (k,)
        Indices of selected coefficients (subset of [0, ..., p]).
    p : int
        Total number of parameters (including intercept).
    """
    _, ax = plt.subplots(figsize=(8, 5))

    # Plot point estimates
    ax.scatter(sel, beta_sel, color="blue", label="Estimand", zorder=3)

    # Plot confidence intervals
    for i, idx in enumerate(sel):
        ax.plot([idx, idx], conf_int[i], color="black", lw=1.5)

    # Add reference line at 0
    ax.axhline(0, color="red", linestyle="--", lw=1)

    ax.set_xticks(range(p))
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_coverage(df, level, x_var="n", type="line"):
    """
    Plot coverage results from a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with columns 'n', 'sparsity', 'method', 'coverage'.
    level : float
        Nominal coverage level (e.g., 0.90).
    x_var : str, optional
        Variable to plot on x-axis
    type : str, optional
        Type of plot: "line" for line plot (coverage vs n),
        "bar" for bar plot (coverage vs sparsity).
    """

    plt.figure(figsize=(7, 5))
    
    if type == "line":
        sns.lineplot(data=df, x=x, y="coverage", hue="method", marker="o")
        plt.xlabel("Sample size n")
    elif type == "bar":
        sns.barplot(data=df, x=x, y="coverage", hue="method")

    plt.axhline(level, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Sparsity")
    plt.ylabel("Coverage")
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.show()
