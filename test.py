import os, subprocess

subprocess.check_call(['pip', 'install', 'dash', 'plotly', 'dash-html-components', 'dash-bootstrap-components'])

import dash
from dash import html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

check = os.environ.get('port')

if check is None:
    check = "0"

details_table = html.Div([
    html.H2("Missile & Drone Model Details"),
    dbc.Button("Reset Sorting", id="reset-sorting-btn", color="primary", outline=True, size="sm"),
    dash_table.DataTable(
        id='details-table',
        columns=['Col1'],
        data=[[check]]
    )
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    dbc.Col(details_table)
])

@app.callback(
        Output('details-table', 'sort_by'),
        Input('reset-sorting-btn', 'n_clicks')
)
def empty_sorting1(sort_by):
    if sort_by == 'hit':
        pass
    else:
        pass

def empty_sorting2(n_clicks):
    if n_clicks:
        return []  # Reset sort_by to an empty list to clear sorting
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)