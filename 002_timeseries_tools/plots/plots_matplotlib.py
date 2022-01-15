from calendar import month_abbr

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def year_over_year_mplot(df, calc, title=None, xlabel="month", ylabel="value", figsize=(10,6)):
    """
    Plots time series year over year. x-axis are months of the year
    """
    df = df.copy()
    columns = df.columns.tolist()
    df['month'] = df.index.strftime('%b')
    df['year'] = df.index.year

    # transform
    if calc=="mean":
        dfp = pd.pivot_table(data=df, index='month', columns='year', values=columns, aggfunc='mean')
    elif calc=="sum":
        dfp = pd.pivot_table(data=df, index='month', columns='year', values=columns, aggfunc='sum')
    else:
        raise ValueError("calc must be either mean or sum")
    
    # the dfp index so the x-axis will be in order
    dfp = dfp.loc[month_abbr[1:]]

    ax = dfp.plot(xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize)
    ax.set_xticks(range(12))  # set ticks for all months
    ax.set_xticklabels(dfp.index)  
    ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    
    return ax

def single_plot_multiple_ts_mplot(df, kind="line", title=None, xlabel=None, ylabel="value", figsize=(10,6)):
    """
    Plots multiple time series into a single plot
    
    kind: line, bar, scatter
    """
    df_copy = df.copy()
    columns = df_copy.columns
    index = df_copy.index.name
    
    df_copy = df_copy.reset_index()
    
    ax = df_copy.plot(x=index, y=columns, xlabel=xlabel, ylabel=ylabel, title=title, kind=kind, backend="matplotlib", figsize=figsize)
    
    plt.show()
    
    return ax

def histogram_grid_mplot(df, n_rows, n_cols, figsize=(12,8)):
    variables = df.columns
    fig=plt.figure(figsize=figsize)
    
    for i, col in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[col].hist(bins=10,ax=ax)
        mean = df[col].mean()
        std = df[col].std()
        ax.set_title(f"{col} mean: {mean:0.2f} dev: {std:0.2f}")
        
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
