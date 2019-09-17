import plotly.express as px
from pandas import DataFrame

from experimentation.results_management.validation.result_objects import ColumnNames


def plot(results: DataFrame):
    results = results[results['agg'] == 'mean']
    fig = px.line(results, x=ColumnNames.QUERY_NUMBER, y="accuracy", color=ColumnNames.STRATEGY)
    fig.show()
    fig = px.line(results, x=ColumnNames.QUERY_NUMBER, y="diversity", color=ColumnNames.STRATEGY)
    fig.show()
    fig = px.line(results, x="diversity", y="accuracy", color=ColumnNames.STRATEGY)
    fig.show()
