from calendar import month_abbr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def year_over_year_mplot(df, calc, title=None, xlabel="value", figsize=(12, 8)):
    df = df.copy()
    columns = df.columns.tolist()
    df["month"] = df.index.strftime("%b")
    df["year"] = df.index.year

    # aggregation
    if calc == "mean":
        dfp = pd.pivot_table(
            data=df, index="month", columns="year", values=columns, aggfunc="mean"
        )
    elif calc == "sum":
        dfp = pd.pivot_table(
            data=df, index="month", columns="year", values=columns, aggfunc="sum"
        )
    else:
        raise ValueError("calc must be either mean or sum")
    cmap = plt.cm.coolwarm

    dfp = dfp.loc[month_abbr[1:]]  # re-order the indexes

    months_dict = dict(enumerate(dfp.index, 1))
    years = dfp.columns.tolist()

    # color map codes for smarter color code. each year has its own color
    cmap_codes = np.linspace(0, 1, len(years))
    cmap_codes = [cmap(x) for x in cmap_codes]
    color_dict = dict(zip(years, cmap_codes))

    height = 0.9 / len(years)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, month in enumerate(months):
        ind = i + 1
        for year in years:
            value = dfp.loc[month, year]
            ax.barh(ind, value, height=0.9 * height, color=color_dict[year])
            if (
                np.isnan(value) == False
            ):  # add value callout using text, only if not nan
                plt.rcParams.update({"font.size": 6})
                ax.text(value, ind, f"{value:.2f}")
                plt.rcParams.update(plt.rcParamsDefault)
            ind = ind + height

    # set y ticks as name of months
    ax.set_yticks(list(months_dict.keys()))
    ax.set_yticklabels(list(months_dict.values()))
    ax.set_xlabel(xlabel)

    if title:
        ax.set_title(title)

    # custom legend
    custom_lines = [Line2D([0], [0], color=cmap_code, lw=4) for cmap_code in cmap_codes]
    ax.legend(custom_lines, years)
    fig = ax.get_figure()
    fig.tight_layout()

    return fig


def single_plot_multiple_ts_mplot(
    df, kind="line", title=None, xlabel=None, ylabel="value", figsize=(10, 6)
):
    """
    Plots multiple time series into a single plot
    
    kind: line, bar, scatter
    """
    df_copy = df.copy()
    columns = df_copy.columns
    index = df_copy.index.name

    df_copy = df_copy.reset_index()

    ax = df_copy.plot(
        x=index,
        y=columns,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        kind=kind,
        backend="matplotlib",
        figsize=figsize,
    )

    plt.show()

    return ax


def histogram_grid_mplot(df, n_rows, n_cols, figsize=(12, 8)):
    variables = df.columns
    fig = plt.figure(figsize=figsize)

    for i, col in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[col].hist(bins=10, ax=ax)
        mean = df[col].mean()
        std = df[col].std()
        ax.set_title(f"{col} mean: {mean:0.2f} dev: {std:0.2f}")
        ax.axvline(mean, color="red")
        ax.axvline(mean + std, color="green")
        ax.axvline(mean - std, color="green")

    fig.tight_layout()
    plt.show()


def auto_corr_plot(ts, lags=None, figsize=(12, 7)):
    """
    Plots time series, auto correlation, partial auto correlation, and performs AD Fuller test
    """
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_vals = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_vals = plt.subplot2grid(layout, (1, 0))
    pacf_vals = plt.subplot2grid(layout, (1, 1))

    ts.plot(ax=ts_vals)
    plot_acf(ts, lags=lags, ax=acf_vals)
    plot_pacf(ts, lags=lags, ax=pacf_vals)
    plt.tight_layout()

    p_value = sm.tsa.stattools.adfuller(ts)[1]
    print(f"Dickey Fuller Test: {p_value:0.5f}")


def subplots_vertical(df):
    """Plots columns of df in subplots stacked vertically"""
    fig, ax = plt.subplots(len(df.columns), 1, figsize=(20, 60))

    for i, col in enumerate(df.columns):
        ax[i].scatter(df[col].index, df[col], c="b")
        ax[i].set_title(f"{col}")

    plt.show()
    return fig
