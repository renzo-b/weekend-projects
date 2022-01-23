from datetime import datetime

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
        dataset = pd.read_excel(input_data, parse_dates=True).set_index("date")
    except ValueError:
        dataset = pd.read_csv(input_data, parse_dates=True,
                              infer_datetime_format=True).set_index("date")
    return dataset


def normalize_columns(x):
    x = x.values
    x = x[~np.isnan(x)]
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(np.array(x).reshape(-1, 1))
    print(np.ravel(x_t))
    print(x)
    return x_t


st.title('Time Series Exploration 📈')
input_data = st.sidebar.file_uploader("", type=[".xlsx", ".csv"])

if input_data:
    task = st.selectbox("Menu", ["Table Exploration", "Signal Analysis"])

    '''Raw dataframe'''
    df = reading_dataset()
    st.dataframe(df)

    raw_or_normalized = st.radio(
        "Normalize described data?", ["raw", "normalized"])
    if raw_or_normalized == "raw":
        st.dataframe(df.describe())
    elif raw_or_normalized == "normalized":
        df_normalized = df.copy()
        for col in df_normalized.columns:
            col_normalized = MinMaxScaler().fit_transform(
                df_normalized[[col]].values)
            df_normalized[col] = col_normalized
        st.dataframe(df_normalized.describe())

    '''Useful stats'''
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

else:
    st.text("To use this tool, just upload a csv file with tabular format")
