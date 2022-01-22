import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_csv(input_data):
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input_data, parse_dates=True, infer_datetime_format=True)
    return df_input


input_data = st.file_uploader("", type=[".csv"])

st.dataframe(input_data)

