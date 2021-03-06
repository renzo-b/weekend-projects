import pandas as pd
import plotly.express as px


def single_plot_single_ts_plotly(ts, kind="line", title=None, xlabel=None, ylabel="value"):
    """
    Plots a single time series into a single plot
    
    kind: line, bar, scatter, box, histogram
    """
    x_vals = ts.index
    y_vals = ts.values
    
    plot_dict = {
        'bar':px.bar(ts, x=x_vals, y=y_vals),
        'box':px.box(ts, y=y_vals),
        'histogram':px.histogram(ts, x=x_vals),
        'scatter': px.scatter(ts, x=x_vals, y=y_vals),
        'line': px.line(ts, x=x_vals, y=y_vals)
    }
    fig = plot_dict[kind]

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=dict(text=title, x=0.5))
    
    fig.show()
    return fig

def single_plot_multiple_ts_plotly(df, kind="line", title=None, xlabel=None, ylabel="value"):
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
    
    fig.show()
    
    return fig
