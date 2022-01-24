from calendar import month_abbr
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from streamlit_pandas_profiling import st_profile_report

input_data = None


def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(input_data, parse_dates=[0]).set_index("date")
    except ValueError:
        dataset = pd.read_csv(
            input_data, parse_dates=[0], infer_datetime_format=True
        ).set_index("date")
    return dataset


def normalize_columns(x):
    x = x.values
    x = x[~np.isnan(x)]
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(np.array(x).reshape(-1, 1))
    print(np.ravel(x_t))
    print(x)
    return x_t


def single_plot_multiple_ts_plotly(
    df, kind="line", title=None, xlabel=None, ylabel="value"
):
    """
    Plots multiple time series into a single plot
    
    kind: line, bar, scatter
    """
    df_copy = df.copy()
    columns = df_copy.columns
    index = df_copy.index.name

    df_copy = df_copy.reset_index()

    fig = df_copy.plot(x=index, y=columns, kind=kind, backend="plotly")

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=dict(text=title, x=0.5))

    return fig


def year_over_year_mplot(
    df, calc, title=None, xlabel="month", ylabel="value", figsize=(10, 6)
):
    """
    Plots time series year over year. x-axis are months of the year
    """
    df = df.copy()
    columns = df.columns.tolist()
    df["month"] = df.index.strftime("%b")
    df["year"] = df.index.year

    # transform
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

    # the dfp index so the x-axis will be in order
    dfp = dfp.loc[month_abbr[1:]]

    ax = dfp.plot(xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize)
    ax.set_xticks(range(12))  # set ticks for all months
    ax.set_xticklabels(dfp.index)
    ax.legend(bbox_to_anchor=(1, 1.02), loc="upper left")

    fig = ax.get_figure()

    return fig


st.title("Time Series Exploration 📈")
input_data = st.sidebar.file_uploader("", type=[".xlsx", ".csv"])

if input_data:
    task = st.selectbox(
        "select the type of analysis",
        ["Useful stats", "Pandas Profiling", "Plotting & Anomalies"],
    )

    """Raw dataframe"""
    df = reading_dataset()
    st.dataframe(df)

    if task == "Useful stats":
        raw_or_normalized = st.radio("Normalize described data?", ["raw", "normalized"])
        if raw_or_normalized == "raw":
            st.dataframe(df.describe())
        elif raw_or_normalized == "normalized":
            df_normalized = df.copy()
            for col in df_normalized.columns:
                col_normalized = MinMaxScaler().fit_transform(
                    df_normalized[[col]].values
                )
                df_normalized[col] = col_normalized
            st.dataframe(df_normalized.describe())

        """Useful stats"""
        useful_stats = {}
        count = df.apply(lambda x: len(x.dropna()))
        missing_values = df.isnull().sum().sort_values(ascending=False)
        duplicates = df.apply(lambda x: x.dropna().duplicated()).sum()
        duplicates_percent = duplicates_percent = duplicates / count * 100

        useful_stats["count"] = count
        useful_stats["Missing values"] = missing_values
        useful_stats["Duplicates"] = duplicates
        useful_stats["Duplicates [%]"] = duplicates_percent

        df_stats = pd.DataFrame(useful_stats)
        st.dataframe(df_stats.style.highlight_quantile(axis=0, q_left=0.8))
    elif task == "Pandas Profiling":
        pr = df.profile_report()
        st_profile_report(pr)
    elif task == "Plotting & Anomalies":
        selected_cols = st.sidebar.multiselect("Pick columns to plot", df.columns)

        col1, col2 = st.columns([1, 3])
        kind = col1.radio("", ["line", "scatter"])
        fig = single_plot_multiple_ts_plotly(
            df[selected_cols], kind=kind, title="Time series plot"
        )
        col2.plotly_chart(fig)

        """ Correlation matrix """
        col1, col2 = st.columns([1, 3])
        corr_columns = col1.radio("", ["all columns", "selected columns"])
        if corr_columns == "all columns":
            corr = df.pct_change().corr()
        elif corr_columns == "selected columns":
            corr = df[selected_cols].pct_change().corr()
        fig = corr.style.background_gradient(cmap="coolwarm").set_precision(2)
        col2.dataframe(fig)

        # agg = st.sidebar.radio("", ["mean", "sum"])
        # fig_2 = year_over_year_mplot(
        #     df[selected_cols], calc=agg, title="year over year plot"
        # )
        # st.pyplot(fig_2)

else:
    st.text("To use this tool, just upload a csv file with tabular format")
