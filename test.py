import os, subprocess

subprocess.check_call(['pip', 'install', 'dash', 'plotly', 'dash-html-components', 'dash-bootstrap-components'])

import dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

check = os.environ.get('port')

btn = dbc.Button("Button", id="button", color="primary", outline=True, size="sm")

if check is None:
    check = "Testing"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    dbc.Row(html.H1(check)),
    dbc.Row(btn)
])

@app.callback(
        Input('button', 'n_clicks')
)
def empty_sorting(n_clicks):
    if n_clicks:
        return []  # Reset sort_by to an empty list to clear sorting
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=10000)