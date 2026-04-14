"""Visualisation utilities for coefficient estimates and simulation coverage."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_selected_coeffs(
    beta_sel,
    conf_int,
    sel,
    p: int,
    title: str = "",
):
    """Plot point estimates and confidence intervals for selected coefficients.

    Parameters
    ----------
    beta_sel : array-like of shape (k,)
        Estimated coefficients for the *k* selected features.
    conf_int : array-like of shape (k, 2)
        Confidence interval ``[lower, upper]`` for each selected coefficient.
    sel : array-like of shape (k,)
        Feature indices of the selected coefficients (subset of
        ``{0, ..., p-1}``).
    p : int
        Total number of parameters (used to set x-axis ticks).
    title : str, optional
        Plot title.
    """
    _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(sel, beta_sel, color="blue", label="Estimate", zorder=3)
    for i, idx in enumerate(sel):
        ax.plot([idx, idx], conf_int[i], color="black", lw=1.5)

    ax.axhline(0, color="red", linestyle="--", lw=1)
    ax.set_xticks(range(p))
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_coverage(df, level: float, x_var: str = "n", kind: str = "line"):
    """Plot empirical coverage from a simulation results dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``'method'``, ``'coverage'``, and *x_var*.
    level : float
        Nominal coverage level (e.g. ``0.90``).  A horizontal reference
        line is drawn at this value.
    x_var : str, default ``'n'``
        Column name to use as the x-axis variable.
    kind : {'line', 'bar'}, default ``'line'``
        Plot style.  ``'line'`` produces a line plot (useful when *x_var*
        is numeric); ``'bar'`` produces a grouped bar plot.
    """
    plt.figure(figsize=(7, 5))

    if kind == "line":
        sns.lineplot(data=df, x=x_var, y="coverage", hue="method", marker="o")
    elif kind == "bar":
        sns.barplot(data=df, x=x_var, y="coverage", hue="method")

    plt.axhline(level, color="red", linestyle="--", linewidth=1, label=f"Nominal ({level:.0%})")
    plt.xlabel(x_var)
    plt.ylabel("Coverage")
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
