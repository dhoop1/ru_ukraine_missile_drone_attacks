import os, subprocess

subprocess.check_call(['pip', 'install', 'dash', 'plotly', 'dash-html-components', 'dash-bootstrap-components'])

check = os.environ.get('port')

import dash, plotly, dash_html_components
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.div([check])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=10000)