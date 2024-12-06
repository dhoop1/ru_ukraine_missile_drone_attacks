# %% [markdown]
# ## Access Kaggle API and import .csv datasets

# %%
#@title ### Access Ukraine missile/UAV attacks datasets

import subprocess

subprocess.check_call(['pip', 'install', 'kaggle', '--quiet'])
subprocess.check_call(['kaggle', 'datasets', 'download', 'piterfm/massive-missile-attacks-on-ukraine', '--force'])

# %%
subprocess.check_call(['pip', 'install', 'zipfile36', '--quiet'])

# %%
#@title ### Unzip datasets (into .csv files)

import zipfile36 as zipfile

with zipfile.ZipFile("massive-missile-attacks-on-ukraine.zip", "r") as file:
    file.extractall("attacks")

# %% [markdown]
# ## Data Preparation

# %%
#@title ### Install modules

subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'tabulate', '--quiet'])

# %%
#@title ### Convert .csv files to DataFrames

import pandas as pd
import numpy as np
import os
from tabulate import tabulate

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:,.1f}".format

# Read 'missile_attacks_daily' csv and create attacks_df
attacks_df_orig = pd.read_csv("attacks/missile_attacks_daily.csv", header=0) # keep as original for comparison/reference
attacks_df = attacks_df_orig.copy() # create as working df

# Fillna() in attacks_df and set float dtypes
numeric_cols = attacks_df.select_dtypes(include=np.number)
attacks_df[numeric_cols.columns] = numeric_cols.fillna(0) # fill NaN with 0 for numeric cols
attacks_df[numeric_cols.columns] = attacks_df[numeric_cols.columns].astype('float') # set numeric cols to float dtypes
non_numeric_cols = attacks_df.select_dtypes(exclude=np.number)
attacks_df[non_numeric_cols.columns] = non_numeric_cols.fillna("") # fill NaN with "" for non-numeric cols

# Read 'missiles_and_uav' csv and create details0_df
details0_df_orig = pd.read_csv("attacks/missiles_and_uav.csv", header=0) # keep as original for comparison/reference
details0_df = details0_df_orig.copy()

model_list = attacks_df["model"].unique()
model_count = attacks_df["model"].value_counts()

# %% [markdown]
# ### Set-up Google Sheets Access

# %%
subprocess.check_call(['pip', 'install', 'gspread', '--quiet'])

# %%
#@title #### Authorize gspread as gc

import gspread

# Authorize gspread with Google Cloud API (per https://docs.gspread.org/en/v6.1.3/oauth2.html#enable-api-access-for-a-project)

credentials = eval(os.environ.get("GOOGLE_KAGGLE_CREDENTIALS"))
gc = gspread.service_account_from_dict(credentials)

# %%
#@title #### Update 'csis_kaggle_ru_ukraine_attacks' worksheet with cleaned 'attacks_df' data

# Select worksheet from 'csis_kaggle_ru_ukraine_attacks' spreadsheet
mad_ref = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('missile_attacks_daily')

# Clear selected worksheet
mad_ref.clear()

# Import 'missile_attacks_daily' (from cleaned attacks_df) into gworksheet
mad_upload = attacks_df.copy()
mad_ref.update([mad_upload.columns.values.tolist()] + mad_upload.values.tolist())

# %% [markdown]
# ## Data-cleaning

# %%
#@title ### Create new columns in attacks_df

# 'model_orig' and new 'model' (blank for now)
if 'model_orig' not in attacks_df.columns:
  attacks_df.rename(columns={'model':'model_orig'}, inplace=True)
  attacks_df['model'] = '' * len(attacks_df.index)

# 'diverged' and 'returned'
attacks_df['diverged'] = attacks_df['not_reach_goal']
attacks_df['returned'] = attacks_df['cross_border_belarus'] + attacks_df['back_russia']

# 'hit' and 'miss'
sub_miss = attacks_df['destroyed'] + attacks_df['diverged'] + attacks_df['returned']
attacks_df['miss'] = sub_miss + (attacks_df['still_attacking'] * (sub_miss / attacks_df['launched'])).round()
attacks_df['hit'] = attacks_df['launched'] - attacks_df['miss']

# 'destroyed_rate' and 'miss_rate'
attacks_df['destroyed_rate'] = (attacks_df['destroyed'] / attacks_df['launched']).round(2)
attacks_df['miss_rate'] = (attacks_df['miss'] / attacks_df['launched']).round(2)

# %%
#@title ### Create simplified cmodel_df (for data-cleaning)

# Original (as source record)
cmodel_df_orig = attacks_df.loc[:, ('time_start', 'time_end', 'model_orig', 'model', 'launched', 'destroyed', 'hit', 'miss', 'destroyed_rate', 'miss_rate')].copy()
#cmodel_df_orig[['launched', 'destroyed', 'not_reach_goal']] = cmodel_df_orig[['launched', 'destroyed', 'not_reach_goal']].astype('float') # Set dtypes for 'launched' and 'destroyed' to float

# Working (for clean-up)
cmodel_df = cmodel_df_orig.copy()

# %% [markdown]
# ### Conduct 'Data Cleaning' on dfs using Google Sheets inputs

# %% [markdown]
# #### Clean-up: Reassign

# %%
#@title ##### Consolidate data for 'Reassign" clean-up

# Create 'reassign_df' with re-formatted index, and drop 'mod_o_count' and 'note' columns
reassign = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('reassign')
reassign_df = pd.DataFrame(reassign.get_all_values(), columns=reassign.row_values(1))
reassign_df = reassign_df.iloc[1:, :-2].reset_index(inplace=False, drop=True) # remove duplicate header row and drop last 2 columns

# Create three sub-df's from the 'reassign' gworksheet
reassign_dfa = reassign_df.iloc[:, [0,1,2,3]].copy() # model_orig entries to update in cmodel, to label with "u" 'alt_code'
reassign_dfb = reassign_df[reassign_df['model_new_2']!=""].iloc[:, [0,4,5,6]] # split entries to insert into cmodel, to label with "i" alt_code
reassign_dfc = reassign_df[reassign_df['model_new_3']!=""].iloc[:, [0,7,8,9]] # more split entries to insert into cmodel, to label with "i" alt_code

# List of df's, and list of new/synchronized column names
rconcat = [reassign_dfa, reassign_dfb, reassign_dfc]
ncols = ['model_orig', 'model', 'alt_type', 'alt', 'alt_code']

# Iterate over the df's, specifying 'alt_code' column and renaming columns
for i, df in enumerate(rconcat):
  if len(df)==len(reassign_dfa):
    df['alt_code']='u'
  else:
    df['alt_code']='i'
  df.columns = ncols

# Concat sub-df's into one df
reassign_dfn = pd.concat(rconcat, ignore_index=True)

# Convert alt column type to float
reassign_dfn[['alt']] = reassign_dfn[['alt']].astype('float')

print(reassign_dfn.columns)

# %%
#@title ##### Clean-up: v3 Reassign (via df logic, using "consolidated" reassign data)

re_ll2 = []

# Iterate over rows in reassign_dfn
for i, row in reassign_dfn.iterrows():

  # Triage with if statements by alt_code

  # Check entry for "i" alt_code, and if so create sub_df (from cmodel_df) to modify and append to re_ll2
  if row.alt_code == "i":
      sub_df = cmodel_df_orig.loc[cmodel_df_orig['model_orig']==row.model_orig].copy() # referencing cmodel_df_orig (to avoid any changes from already looped cmodel_df entries)
      sub_df.loc[sub_df['model_orig']==row.model_orig, 'model'] = row.model

      if row.alt_type == "*=": # just a formality
        sub_df.loc[sub_df['model_orig']==row.model_orig, ['launched', 'destroyed', 'hit', 'miss', 'miss_rate', 'destroyed_rate']] *= row.alt
        sub_df[['launched', 'destroyed', 'hit', 'miss']] = sub_df[['launched', 'destroyed', 'hit', 'miss']].round(0)
        sub_df[['miss_rate', 'destroyed_rate']] = sub_df[['miss_rate', 'destroyed_rate']].round(2)

      re_ll2.append(sub_df)

  # Then check entry for "u" alt_code, and if so update cmodel_df according to alt_type and l_/d_alt
  if row.alt_code == "u":
      cmodel_df.loc[cmodel_df['model_orig']==row.model_orig, 'model'] = row.model

      if row.alt_type == "*=": # just a formality
        cmodel_df.loc[cmodel_df['model_orig']==row.model_orig, ['launched', 'destroyed', 'hit', 'miss', 'miss_rate', 'destroyed_rate']] *= row.alt
        cmodel_df[['launched', 'destroyed', 'hit', 'miss']] = cmodel_df[['launched', 'destroyed', 'hit', 'miss']].round(0)
        cmodel_df[['miss_rate', 'destroyed_rate']] = cmodel_df[['miss_rate', 'destroyed_rate']].round(2)

# %%
#@title ##### Convert re_ll2 to re_df2 and concat to cmodel_df

# Drop indices from dfs in re_ll2
for dfr in re_ll2:
    dfr = dfr.reset_index(drop=True, inplace=True)

# Concat re_ll2 into re_df2
re_df2 = pd.concat(re_ll2)

# Concat re_df2 into cmodel_df
cmodel_df = pd.concat([cmodel_df, re_df2], ignore_index=True)

print("Concat complete.")

# %% [markdown]
# #### Clean-up: Aggregate

# %%
# Create agg_df from gworksheet 'aggregate', and drop 'note' column
agg = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('aggregate')
agg_df = pd.DataFrame(agg.get_all_values(), columns=agg.row_values(1))
agg_df = agg_df.iloc[1:, :-1].reset_index(inplace=False, drop=True) # remove duplicate header row and drop last column

for ia, rowa in agg_df.iterrows():
    cmodel_df.loc[cmodel_df['model_orig']==rowa.model_orig_1, 'model'] = rowa.model
    cmodel_df.loc[cmodel_df['model_orig']==rowa.model_orig_2, 'model'] = rowa.model

# %% [markdown]
# #### Clean-up: Modify v2

# %%
# Create mdfy_df from gworksheet 'modify', drop last three columns, and update dtypes
mdfy = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('modify')
mdfy_df = pd.DataFrame(mdfy.get_all_values(), columns=mdfy.row_values(1))
mdfy_df = mdfy_df.iloc[1:, :-6].reset_index(inplace=False, drop=True) # remove duplicate header row and drop last six columns
mdfy_df[['launched', 'destroyed', 'hit', 'miss']] = mdfy_df[['launched', 'destroyed', 'hit', 'miss']].astype('float') # set mdfy_df 'launched' and 'destroyed' dtypes to float

# Create empty list (for sub_dfs to append)
m_ll = []

# Iterate over rows in mdfy_df per conditions and perform following actions
for im, rowm in mdfy_df.iterrows():

  if rowm.modified == "1": # 'model' has already been assigned and can be referenced
      cmodel_df.loc[(cmodel_df['time_end']==rowm.time_end) & (cmodel_df['model']==rowm.model), ['launched', 'destroyed', 'hit' ,'miss']] = [rowm.launched, rowm.destroyed, rowm.hit, rowm.miss]
      cmodel_df.loc[(cmodel_df['time_end']==rowm.time_end) & (cmodel_df['model']==rowm.model), ['miss_rate', 'destroyed_rate']] = [rowm.miss / rowm.launched, rowm.destroyed / rowm.launched] # split out from above for readability

  if rowm.modified == "0": # 'model' not yet assigned, thus reference 'model_orig'
      if rowm.alt_code == "i": # create and update subm_df to append/insert into cmodel_df
          subm_df = cmodel_df.loc[(cmodel_df['time_end']==rowm.time_end) & (cmodel_df['model_orig']==rowm.model_orig)].copy()
          subm_df[['model', 'launched', 'destroyed', 'hit' ,'miss']] = [rowm.model, rowm.launched, rowm.destroyed, rowm.hit, rowm.miss]
          sub_df[['miss_rate', 'destroyed_rate']] = [rowm.miss / rowm.launched, rowm.destroyed / rowm.launched] # split out from row above for readability
          m_ll.append(subm_df)
      elif rowm.alt_code == "u": # update cmodel_df
          cmodel_df.loc[(cmodel_df['time_end']==rowm.time_end) & (cmodel_df['model_orig']==rowm.model_orig), ['launched', 'destroyed', 'hit' ,'miss']] = [rowm.launched, rowm.destroyed, rowm.hit, rowm.miss]
          cmodel_df.loc[(cmodel_df['time_end']==rowm.time_end) & (cmodel_df['model_orig']==rowm.model_orig), ['miss_rate', 'destroyed_rate']] = [rowm.miss / rowm.launched, rowm.destroyed / rowm.launched]

# Convert and concat/append mdfy_ll into cmodel_df
for dfm in m_ll:
    dfm = dfm.reset_index(drop=True, inplace=True)

m_df = pd.concat(m_ll)

cmodel_df = pd.concat([cmodel_df, m_df], ignore_index=True)

print("Actions and concat complete.")

# %% [markdown]
# ### Clean-up: Final steps

# %%
#@title ### Clean-up: Add 'model' for unmodified entries

modified_entries = cmodel_df.loc[cmodel_df['model']!=""].index
model_check = cmodel_df.loc[cmodel_df['model']!="", 'model'].copy()

cmodel_df.loc[cmodel_df['model']=="", 'model'] = cmodel_df.loc[cmodel_df['model']=="", 'model_orig']

if cmodel_df.loc[modified_entries, 'model'].to_list() == model_check.to_list():
  print("Successfully added 'model' to other entries.")
else:
  print("Possible error and/or over-writing 'model' of cleaned entries. Check data.")

# %%
#@title ### Clean-up: Error Checking (via comparing 'cmodel_df_orig' and 'cmodel_df')

print("'cmodel_df_orig' summed counts:")
print(cmodel_df_orig[['launched', 'destroyed']].sum(), "\n")
print("'cmodel_df' summed counts:")
print(cmodel_df[['launched', 'destroyed']].sum(), "\n")
print("Difference in summed counts:")
print(cmodel_df_orig[['launched', 'destroyed']].sum() - cmodel_df[['launched', 'destroyed']].sum(), "\n")

# Create variables for total, launched, and destroyed differentials
l_diff, d_diff = cmodel_df_orig[['launched', 'destroyed']].sum() - cmodel_df[['launched', 'destroyed']].sum().to_list()

# Create check_df as the difference betwen cmodel_df_orig and cmodel_df, grouped by model_orig
check_df = cmodel_df_orig.groupby('model_orig')[['launched', 'destroyed']].sum() - cmodel_df.groupby('model_orig')[['launched', 'destroyed']].sum()

# Displays model_orig rows (if any) for which subracted 'launched' or 'destroyed' is not 0 -- no entry should deviate by ~5 or more, else indicates a potential error
err_comp_df = check_df[(check_df['launched'] != 0) | (check_df['destroyed'] != 0)]
print("Difference between cmodel_df_orig and cmodel_df, by 'model_orig':")
print(tabulate(err_comp_df[0:], headers=err_comp_df.columns, tablefmt='simple_outline'))

print() # blank line
print("These discrepances are known and accepted:")
print("C-400 and Iskander-M	7.0	7.0 -- correction from data capture error in Kaggle dataset")
print("Iskander-M/KN-23	-1.0	3.0 -- due to rounding during clean-up")
print("Orlan-10 and ZALA	-1.0	-1.0 -- due to rounding during clean-up")
print("X-101/X-555 and Kalibr and X-59/X-69 -6.0	-5.0 -- due to data correction as specified in Kaggle dataset.")
print() # blank line

num_known_discrep = 4    # manually entered

print("Summary:")
if len(err_comp_df) == num_known_discrep:
  print("Data clean-up seems successful (same number of discrepancies as previously identified).")
else:
  print("Changes have resulted since error-checking last reviewed; it is recommended to review again.")

# %%
#@title #### Checking newly created column totals ('hit' and 'miss')

print(tabulate(cmodel_df[(cmodel_df['hit'] + cmodel_df['miss']) > cmodel_df['launched']], headers=cmodel_df.columns, tablefmt='simple_outline'))

print() # blank line
print("These discrepancies are known and accounted:")
print("model_orig   model  launched  destroyed hit miss")
print("X-101/X-555 and Kalibr	Kalibr	43.0	37.0	7.0	37.0")

# %% [markdown]
# ### Insert model_detail data from Google Sheets

# %%
#@title #### Convert 'ru_model_detail' worksheet into a df

detail = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('model_detail')
detail_df = pd.DataFrame(detail.get_all_values(), columns=detail.row_values(1))
detail_df = detail_df.iloc[1:, :].reset_index(inplace=False, drop=True)

if detail_df.isna().sum().sum() == 0:
  print("Complete (no NA entries).")
else:
  print("Check upload datasheet--NA entries present.")

# %%
#@title #### Merge cmodel_df and detail_df with an inner join

# prompt: Join on inner cmodel_df and detail_df, so that each indexed row in cmodel_df has columns added that mapped over from detail_df.

merged_df = pd.merge(cmodel_df, detail_df, on='model', how='inner')

merged_df['miss_rate'] = merged_df['miss_rate'].round(2)

merged_df

# %% [markdown]
# ## Data Analytics

# %%
#@title ### Descriptive analytics on cmodel_df

from tabulate import tabulate

print(tabulate(merged_df.groupby(['category'])[['launched', 'hit', 'miss', 'destroyed']].sum().sort_values('launched', ascending=False), headers='keys', tablefmt='simple_outline', floatfmt=",.0f"))

# %% [markdown]
# ### DPX Components

# %%
#@title Install dash/plotly modules

subprocess.check_call(['pip', 'install', 'dash', 'plotly', 'dash-html-components', 'dash-core-components', 'dash-bootstrap-components', '--quiet'])

# %% [markdown]
# #### DPX app: ts_bar_chart

# %%
#@title #### ts_bar_chart

# prompt: From merged_df, create a Dash Plotly time series visualization with a stacked bar chart, and the data determined by the following radio button selections. The x-axis will be time as either week, month, quarter, or year selected by a radio button (default is month). The y-axis will be 'launched', 'hit', 'miss', or 'destroyed' as selected by a radio button (default is 'launched'). The stacks in the bar chart will be either 'category' or 'model' as selected by a radio button (default is 'category').

import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Assuming merged_df is already defined from the previous code

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

ts_bar_chart = html.Div([

    # title
    dbc.Row([html.H2("Time Series for Missile and Drone Attacks Against Ukraine")]),

    # first row of radio buttons
    dbc.Row([

        # time_period
        dbc.Col([
            dbc.Row(html.Label("Time Period:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
            dbc.RadioItems(
                id='time-period',
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {'label': 'Week', 'value': 'W'},
                    {'label': 'Month', 'value': 'M'},
                    {'label': 'Quarter', 'value': 'Q'},
                    {'label': 'Year', 'value': 'Y'}
                ],
                value='M'  # Default value
            )
        ]),

        # yaxis
        dbc.Col([
            dbc.Row(html.Label("Y-axis:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
            dbc.RadioItems(
                id='y-axis-value',
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {'label': 'Launched', 'value': 'launched'},
                    {'label': 'Hit', 'value': 'hit'},
                    {'label': 'Miss', 'value': 'miss'},
                    {'label': 'Destroyed', 'value': 'destroyed'}
                ],
                value='launched' # Default value
            )
        ]),

        # stacked_bar_feature
        dbc.Col([
            dbc.Row(html.Label("Stacked Bar Feature:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
            dbc.RadioItems(
                id='stacked-bar-feature',
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {'label': 'Category', 'value': 'category'},
                    {'label': 'Model', 'value': 'model'}
                ],
                value='category' # Default value
            )
        ]),

        # yaxis2
        dbc.Col([
            dbc.Row(html.Label("Second Y-Axis:", style={'textAlign': 'left', 'fontWeight': 'bold', 'margin-bottom':1})),
            dbc.Checklist(
                id='y-axis2-toggle',
                options=[{'label': 'Launched', 'value': 'launched'}],
                value=[],
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active"
            )
        ])
    ]),


    # second row of radio buttons
    dbc.Row(html.Div([
        html.H6("Category:", style={'fontWeight': 'bold', 'display': 'inline-block', 'padding-top': '5px'}),
        dbc.Checklist(
            id='category-slice',
            options=[{'label': category, 'value': category} for category in merged_df['category'].unique()],
            value=merged_df['category'].unique().tolist(),  # Default: all categories selected
            inline=True,  # Arrange options horizontally
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active"
        ),
    ], style={'padding-top': '5px', 'textAlign': 'center'})),

    # third/final row with ts_bar_chart
    dbc.Row([dcc.Graph(id='time-series-chart')])

])

# %% [markdown]
# ##### ts_bar_chart @app.callback

# %%
#@title ts_bar_chart @app.callback

#@app.callback(
#    Output('time-series-chart', 'figure'),
#    Input('time-period', 'value'),
#    Input('y-axis-value', 'value'),
#    Input('stacking-category', 'value'),
#    Input('y-axis2-toggle', 'value')
#)
def update_chart(time_period, y_axis_value, stacking_category, y_axis2):
    # Resample the data based on the selected time period
    temp_df3 = merged_df.copy()
    temp_df3['time_end'] = pd.to_datetime(merged_df['time_end'], format="mixed")
    temp_df3['time_period'] = temp_df3['time_end'].dt.to_period(time_period)
    temp_df3['time_period'] = temp_df3['time_period'].astype(str)

    # 1. Generate complete time range
    all_periods = pd.period_range(
        start=temp_df3['time_end'].min(),
        end=temp_df3['time_end'].max(),
        freq=time_period
    ).astype(str).tolist()

    # Group and sum data
    grouped_data = temp_df3.groupby(['time_period', stacking_category])[y_axis_value].sum().reset_index()

    # 2. Reindex and fill missing values
    # Create a MultiIndex from all_periods and unique stacking_category values
    all_categories = grouped_data[stacking_category].unique()
    full_index = pd.MultiIndex.from_product([all_periods, all_categories], names=['time_period', stacking_category])

    # Reindex the grouped data
    reindexed_data = grouped_data.set_index(['time_period', stacking_category]).reindex(full_index, fill_value=0).reset_index()

    # 3. Create the stacked bar chart
    fig = px.bar(
        reindexed_data,
        x='time_period',
        y=y_axis_value,
        color=stacking_category,
        title=f"{y_axis_value.capitalize()} by {stacking_category.capitalize()} over Time",
        barmode="stack"
    )

    fig.update_layout(
        font_color="gray",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Adjust this value to fine-tune the legend's position
            xanchor="center",
            x=0.5  # Adjust this value to fine-tune the legend's position
        )
    )

    # 4. Create a line chart from the second axis for 'launched' (if y_axis_value not 'launched')
    if 'launched' in y_axis2:

        launched_data = temp_df3.groupby('time_period')['launched'].sum().reset_index()
        #yaxis2_range = [0, launched_data['launched'].max()]

        fig.add_scatter(
            x=launched_data['time_period'],
            y=launched_data['launched'],
            mode='lines',
            name='launched',
            yaxis='y2',
            line=dict(color='red')
        )

        fig.update_layout(
            yaxis2=dict(
                title='launched',
                title_font=dict(color='red'),
                overlaying='y',
                side='right',
                matches='y',
                #range=yaxis2_range,
                showgrid=False,
                #gridcolor='gray',
                #griddash='dot',
                tickfont=dict(color='red'),
            ),
        )

    # 5. Adjust legend

    # Remove if time_period is 'W'
    if time_period == 'W':
        fig.update_layout(
            showlegend=False
        )

    # Space further down on y-axis if Stacked Bar Feature is 'model'
    if stacking_category == 'model':
        fig.update_layout(
            legend=dict(
                y=-0.4
            )
        )

    return fig

#app.layout = html.Div([
#    dbc.Row([ts_bar_chart]),
#    dbc.Row([details_table])
#    ])

#if __name__ == '__main__':
    #app.run_server(debug=True)
#    app.run_server(mode='jupyterlab', debug=True)

# %% [markdown]
# #### DPX app: details_table

# %%
#@title DPX app: details_table

# prompt: Using Dash Plotly and from merge_df, create a table that groups merged_df by the columns from detail_df (i.e. detail_df.columns) and aggregates 'launched', 'hit', 'miss', 'destroyed', and 'miss_rate' the rows.

import dash
from dash import dash_table, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# Create the data for the PX table, using df.groupby().agg()
details_data = merged_df.groupby(detail_df.columns.tolist())[['launched', 'hit', 'miss', 'destroyed', 'miss_rate']].agg({
            'launched': 'sum',
            'hit': 'sum',
            'miss': 'sum',
            'destroyed': 'sum',
            'miss_rate': 'mean'
        }).reset_index()

# Sort the details_data into category and type, descending by max 'launched' value per category and type, then by 'launched' within each 'category-type'
details_data['miss_rate'] = details_data['miss_rate'].round(2)
details_data['destroyed_rate'] = (details_data['destroyed'] / details_data['launched']).round(2)
details_data['max_l_cat'] = details_data.groupby(['category'])['launched'].transform('max')
details_data['max_l_type'] = details_data.groupby(['type'])['launched'].transform('max')
details_data = details_data.sort_values(by=['max_l_cat', 'max_l_type', 'launched'], ascending=False, inplace=False)
details_data.drop(columns=['max_l_cat', 'max_l_type'], inplace=True) # drop temporary columns used for sorting

# Create variable of column names to use in detail_table
details_columns = [{"name": i, "id": i} for i in detail_df.columns.tolist() + ['launched', 'hit', 'miss', 'destroyed', 'miss_rate']]

# PX table data and html formatting
details_table = html.Div([
    html.H2("Missile & Drone Model Details"),
    dbc.Button("Reset Sorting", id="reset-sorting-btn", color="primary", outline=True, size="sm"),
    dash_table.DataTable(
        id='details-table',
        columns=details_columns,
        data=details_data.to_dict('records'),
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'lightgray',
            'fontWeight': 'bold',
            'whitespace': 'normal',
            'height': 'auto',
            #'minwidth': '20px'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'lightgray',
        },
        style_data_conditional=[
            {'if': {'column_id': 'miss_rate'}, 'type': 'float', 'format': {'specifier': '.2f'}} # Format with two decimal places
        ],
        style_cell={'fontSize': 8, 'minwidth': '4%'},
        style_cell_conditional=[
            {'if': {'column_id': ['model', 'type', 'detail', 'maneuverability']}, 'width': '8%'},
            {'if': {'column_id': ['category', 'range', 'altitude', 'speed']}, 'width': '7%'},
            {'if': {'column_id': ['payload', 'accuracy', 'unit_cost_usd']}, 'width': '5.33%'},
            {'if': {'column_id': ['launched', 'hit', 'miss', 'destroyed', 'miss_rate', 'destroyed_rate']}, 'width': '4%'},
        ],
        #style_table={'overflowX': 'auto'},
        sort_action='native',
        sort_mode='multi',
        page_size=10,
        #style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
    )
])

# %% [markdown]
# ##### details_table @app.callback

# %%
#@app.callback(
#    Output('details-table', 'sort_by'),
#    Input('reset-sorting-btn', 'n_clicks'),
#    #State('details-table', 'sort_by') # the 'Reset Sorting' button doesn't work with State active
#)
def reset_sorting(n_clicks):
    if n_clicks:
        return []  # Reset sort_by to an empty list to clear sorting
    return dash.no_update  # Don't update if button hasn't been clicked

#app.layout = html.Div([details_table])

#if __name__ == '__main__':
#    app.run_server(debug=True)

# %% [markdown]
# #### DPX app: attacks_table

# %%
#@title attacks_table

# prompt: Using Dash Plotly and from merged_df, create a table which displays the 'launched', 'hit', 'miss', and 'destroyed' summed-up counts, grouped by 'category'. Title this table as, "Missile Attacks by Category, for All Time".

from dash import dash_table, html
from dash import dash_table
from dash.dash_table.Format import Format, Group, Scheme

# Assuming merged_df is already defined from the previous code

attacks_table = html.Div([
    html.H4("Missile Attacks by Category"),
    dash_table.DataTable(
        id='attacks-table',
        #columns=[{"name": i, "id": i} for i in ['category', 'launched', 'hit', 'miss', 'destroyed']],
        columns=[
            {"name": 'category', "id": 'category', "type": 'text'},  # Set 'category' to string/object type
            {"name": 'launched', "id": 'launched', "type": 'numeric', "format": Format().group(True)},  # Set 'launched' to numeric
            {"name": 'hit', "id": 'hit', "type": 'numeric', "format": Format().group(True)},  # Set 'hit' to numeric
            {"name": 'miss', "id": 'miss', "type": 'numeric', "format": Format().group(True)},  # Set 'miss' to numeric
            {"name": 'destroyed', "id": 'destroyed', "type": 'numeric', "format": Format().group(True)}  # Set 'destroyed' to numeric
        ],
        data=merged_df.groupby('category')[['launched', 'hit', 'miss', 'destroyed']].sum().sort_values('hit', ascending=False).reset_index().to_dict('records'),
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'lightgray',
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'lightgray',
        },
        style_table={
            'overflowX': 'auto',
            'width': '100%',
        },
        style_cell={'fontSize': 14, 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'category'}, 'width': '32%'}, # Set width for 'category'
            {'if': {'column_id': ['launched', 'hit', 'miss', 'destroyed']}, 'width': '17%', 'format': {'specifier': ',.0f'}}  # Equal width for other columns, and center-aligned text
        ]
    )
])

# %% [markdown]
# #### DPX app: top_models_table

# %%
#@title top_models_table

# prompt: Create a Dash Plotly Express data table from merge_df with the columns 'model', 'launched', 'hit', 'miss', 'miss_rate', 'destroyed', and 'destroyed_rate', which shows ten models with the highest 'hit value', lowest 'miss_rate', or highest 'destroyed_rate' (using radio buttons to switch between 'hit', 'miss_rate', and 'destroyed_rate'). Format the radio buttons using dash-bootstrap-components.

import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dash_table.Format import Format, Group, Scheme

# Aggregate data for top_models table from merged_df, and add 'hit_rate' column
tm_df = merged_df.groupby('model').agg(
    launched=('launched', 'sum'),
    hit=('hit', 'sum'),
    miss=('miss', 'sum'),
    destroyed=('destroyed', 'sum'),
    miss_rate=('miss_rate', 'mean'),
    destroyed_rate=('destroyed_rate', 'mean')
).reset_index()
tm_df['hit_rate'] = 1 - tm_df['miss_rate']

tm_df['hit_rate'] = (1 - tm_df['miss_rate']).round(2)
tm_df['miss_rate'] = tm_df['miss_rate'].round(2)
tm_df['destroyed_rate'] = tm_df['destroyed_rate'].round(2)

# Define app and stylesheet/theme
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# Define top_models_table with radio buttons
top_models_table = html.Div([
    html.H4("Top-striking Models"),
    dbc.RadioItems(
        id='sort-by-radio',
        options=[
            {'label': 'Hit (max)', 'value': 'hit'},
            {'label': 'Hit Rate (max)', 'value': 'hit_rate'},
            {'label': 'Destroyed Rate (min)', 'value': 'destroyed_rate'}
        ],
        value='hit',  # Default sorting
        inline=True
    ),
    dash_table.DataTable(
        id='top-models-table',
        columns=[
            dict(id='model', name='model'),
            dict(id='launched', name='launched', type='numeric', format=Format().group(True)),
            dict(id='hit', name='hit', type='numeric', format=Format().group(True)),
            dict(id='hit_rate', name='hit_rate', type='numeric', format=Format(precision=0, scheme=Scheme.percentage)), # format added
            dict(id='miss', name='miss', type='numeric', format=Format().group(True)),
            dict(id='miss_rate', name='miss_rate', type='numeric', format=Format(precision=0, scheme=Scheme.percentage)), # format added
            dict(id='destroyed', name='destroyed', type='numeric', format=Format().group(True)),
            dict(id='destroyed_rate', name='destroyed_rate', type='numeric', format=Format(precision=0, scheme=Scheme.percentage)) # format added
        ],
        data=[],  # Initialize with empty data
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'lightgray'
        },
        style_table={
            'overflowX': 'auto',
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'lightgray'
        },
        style_cell={'fontSize': 14, 'textAlign': 'center', 'className': 'pr-2'},
    )
])

# %% [markdown]
# ##### top_models_table @app.callback()

# %%
#@app.callback(
#    Output('top-models-table', 'data'),
#    Input('sort-by-radio', 'value')
#)
def update_top_models_table(sort_by):
    if sort_by == 'hit':
        top_models = tm_df.nlargest(10, 'hit')
    elif sort_by == 'hit_rate':
        top_models = tm_df.nlargest(10, 'hit_rate')
    elif sort_by == 'destroyed_rate':
        top_models = tm_df.nsmallest(10, 'destroyed_rate')
    return top_models.to_dict('records')

#app.layout = html.Div([top_models_table])

#if __name__ == '__main__':
#    app.run_server(debug=True)

# %% [markdown]
# #### DPX app: intro_info

# %%
#@title intro_info

# brief description and image

# Website: "https://www.kaggle.com/code/piterfm/massive-missile-attacks-on-ukraine-data-review/notebook"
# Facebook: https://facebook.com/kpszsu

intro_text = dbc.Card(
    dbc.CardBody([
        html.H4("Description", className="card-title"),
        html.P([
            "This dashboard visualizes data on Russian missile and drone attacks in Ukraine. The data source is the Kaggle dataset ",
            html.A("Massive Missile Attacks on Ukraine", href="https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine", target="_blank"),
            " which sources all data from the Ukrainian Air Force's social media (Facebook, Twitter, Telegram, etc.) and other established sources (e.g., 'war_monitor' Telegram)."
        ], className="card-text", style={'fontSize': '12px'},
        ),
        html.Div(html.Img(src="https://pbs.twimg.com/profile_images/1630572076045152256/FoDWY513_400x400.jpg", alt='kpszsu_logo', style={'width': '70%'}), style={'textAlign': 'center'}),
        html.P(["Source: ", html.A("KpsZSU", href="https://x.com/KpsZSU/photo", target="_blank")], style={'fontSize': '8px', 'textAlign': 'center'})
    ]),
    #style={'border': '1px solid lightgray', "margin-bottom":"15px"},
    style={'border': 'none', 'background-color': 'transparent'}
)

# %% [markdown]
# #### DPX app: metrics_bar

# %%
#@title metrics_table (data)

# days_at_war
from datetime import date
min_date = pd.to_datetime(merged_df['time_start'], format="mixed").min()
today = date.today()
days_at_war = (today - min_date.date()).days

# total_launched
total_launched = merged_df['launched'].sum()

# total_ru_mad_cost
merged_df['unit_cost_usd'] = merged_df['unit_cost_usd'].astype(str).str.replace(",", "", regex=False)
merged_df['unit_cost_usd'] = pd.to_numeric(merged_df['unit_cost_usd'], errors='coerce')
attack_cost = merged_df['launched'] * merged_df['unit_cost_usd'].astype(float)
total_ru_mad_cost = (attack_cost.sum()/1000000000).round(1)

# last_30days_launched
merged_df['time_end'] = pd.to_datetime(merged_df['time_end'], format="mixed")
last_30days_launched = merged_df[merged_df['time_end'] >= (pd.Timestamp(today) - pd.Timedelta(days=30))]['launched'].sum()

# %%
#@title v2 metrics_table (html.Div)

# prompt: Create metrics_bar2 = html.Div() which contains days_at_war, total_launched, total_ru_mad_cost, and last_30days_launched as four Cards horizontally adjacent (which "share" a common background and outline), with the descriptor in html.H6() and the variable in html.P() smaller font size.

metrics_data = [{
        'days-at-war': days_at_war,  # Reference actual variable
        'total-cost': total_ru_mad_cost,  # Reference actual variable
        'total-launched': total_launched,  # Reference actual variable
        'last-30days-launched': last_30days_launched  # Reference actual variable
    }]

metrics_bar4 = html.Div([
    html.H4("Summary Metrics"), # style={'textAlign': "center"}
    dbc.Row([
        dash_table.DataTable(
            id='metrics-table',
            columns=[
                dict(id='days-at-war', name="Days \n at War", type='numeric', format=Format().group(True)),
                dict(id='total-cost', name="Total Cost (US$B)", type='numeric', format=Format(precision=1, scheme=Scheme.fixed)),
                dict(id='total-launched', name="Total \n Launched", type='numeric', format=Format().group(True)),
                dict(id='last-30days-launched', name="Last 30 Days Launched", type='numeric', format=Format().group(True)),
            ],
            data=metrics_data,
            style_header={
                'backgroundColor': 'transparent',
                'color': 'lightgray',
                'fontWeight': 'bold',
                'font-size': 14,
                'border': 'none',
                'textAlign': 'center',
                'whiteSpace': 'pre-line',
            },
            style_table={
                'overflowX': 'auto',
                'border': 'none',
                #'border-collapse': 'collapse',
                'margin': 4,
                'padding': '2% 1% 2% 0%'
            },
            style_data={
                'backgroundColor': 'transparent',
                'color': 'lightgray',
                'border': 'none',
            },
            style_cell={
                'fontSize': 14,
                'textAlign': 'center',
                'width': '25%',
                'border': 'none',
            },
            style_header_conditional=[
                {'if': {'column_id': ['total-cost', 'last-30days-launched']}, 'background-color': 'rgba(211, 211, 211, 0.2)'},
                {'if': {'column_id': ['days-at-war', 'total-launched']}, 'background-color': 'rgba(211, 211, 211, 0.1)'}
            ],
            style_data_conditional=[
                {'if': {'column_id': ['total-cost', 'last-30days-launched']}, 'background-color': 'rgba(211, 211, 211, 0.1)'},
                #{'if': {'column_id': ['days-at-war', 'total-launched']}, 'background-color': 'rgba(211, 211, 211, 0.05)'}
            ],
        )
    ]) # style={'border': '1px solid lightgray', 'margin': 4, 'padding': '2%'}
])

# %% [markdown]
# #### DPX app and @app.callback (combined)

# %%
#@title DPX app + @app.callback

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# ---------------------------------------------------------------------------

# top_models_table
@app.callback(
    Output('top-models-table', 'data'),
    Input('sort-by-radio', 'value')
)
def update_top_models_table(sort_by):
    if sort_by == 'hit':
        top_models = tm_df.nlargest(10, ['hit', 'hit_rate'])
    elif sort_by == 'hit_rate':
        top_models = tm_df.nlargest(10, ['hit_rate', 'hit'])
    elif sort_by == 'destroyed_rate':
        top_models = tm_df.nsmallest(10, ['destroyed_rate', 'hit'])
    return top_models.to_dict('records')


# ---------------------------------------------------------------------------

# ts_bar_chart
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('time-period', 'value'),
    Input('y-axis-value', 'value'),
    Input('stacked-bar-feature', 'value'),
    Input('y-axis2-toggle', 'value'),
    Input('category-slice', 'value')
)
def update_chart(time_period, y_axis_value, stacked_bar_feature, y_axis2, category_slice):
    # Resample the data based on the selected time period
    temp_df3 = merged_df.copy()
    temp_df3['time_end'] = pd.to_datetime(merged_df['time_end'], format="mixed")
    temp_df3['time_period'] = temp_df3['time_end'].dt.to_period(time_period)
    temp_df3['time_period'] = temp_df3['time_period'].astype(str)

    # 1. Generate complete time range
    all_periods = pd.period_range(
        start=temp_df3['time_end'].min(),
        end=temp_df3['time_end'].max(),
        freq=time_period
    ).astype(str).tolist()

    # Slice data (by category_slice) and groupby/sum (by stacked_bar_feature)
    sliced_data = temp_df3[temp_df3['category'].isin(category_slice)].reset_index()
    grouped_data = sliced_data.groupby(['time_period', stacked_bar_feature])[y_axis_value].sum().reset_index()

    # 2. Reindex and fill missing values
    # Create a MultiIndex from all_periods and unique stacking_category values
    all_categories = grouped_data[stacked_bar_feature].unique()
    full_index = pd.MultiIndex.from_product([all_periods, all_categories], names=['time_period', stacked_bar_feature])

    # Reindex the grouped data
    reindexed_data = grouped_data.set_index(['time_period', stacked_bar_feature]).reindex(full_index, fill_value=0).reset_index()

    # 3. Create the stacked bar chart
    fig = px.bar(
        reindexed_data,
        x='time_period',
        y=y_axis_value,
        color=stacked_bar_feature,
        title=f"{y_axis_value.capitalize()} by {stacked_bar_feature.capitalize()} over Time",
        barmode="stack"
    )

    fig.update_layout(
        font_color="gray",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Adjust this value to fine-tune the legend's position
            xanchor="center",
            x=0.5  # Adjust this value to fine-tune the legend's position
        )
    )

    # 4. Create a line chart from the second axis for 'launched' (if y_axis_value not 'launched')
    if 'launched' in y_axis2:

        launched_data = temp_df3.groupby('time_period')['launched'].sum().reset_index()
        #yaxis2_range = [0, launched_data['launched'].max()]

        fig.add_scatter(
            x=launched_data['time_period'],
            y=launched_data['launched'],
            mode='lines',
            name='launched',
            yaxis='y2',
            line=dict(color='red')
        )

        fig.update_layout(
            #yaxis=dict(
            #    range=yaxis2_range
            #),
            yaxis2=dict(
                title='launched',
                title_font=dict(color='red'),
                overlaying='y',
                side='right',
                matches='y',
                #range=yaxis2_range,
                showgrid=False,
                #gridcolor='gray',
                #griddash='dot',
                tickfont=dict(color='red'),
            ),
        )

    # 5. Adjust legend

    # Remove if time_period is 'W'
    if time_period == 'W':
        fig.update_layout(
            showlegend=False
        )

    # Space further down on y-axis if Stacked Bar Feature is 'model'
    if stacked_bar_feature == 'model':
        fig.update_layout(
            legend=dict(
                y=-0.4
            )
        )

    return fig

# ---------------------------------------------------------------------------

# details_table
@app.callback(
    Output('details-table', 'sort_by'),
    Input('reset-sorting-btn', 'n_clicks'),
    #State('details-table', 'sort_by') # the 'Reset Sorting' button doesn't work with State active
)
def reset_sorting(n_clicks):
    if n_clicks:
        return []  # Reset sort_by to an empty list to clear sorting
    return dash.no_update  # Don't update if button hasn't been clicked


# ---------------------------------------------------------------------------

ru_ua_title = "Russian Missile and Drone Attacks against Ukraine"

# app.layout
app.layout = html.Div([
    html.H1(ru_ua_title),
    dbc.Row([
        dbc.Col([intro_text], width=2),
        dbc.Col([
            dbc.Row([metrics_bar4], className='pt-0 pb-6 pl-2 pr-2'),
            dbc.Row([attacks_table], className='pt-3'),
            ], width=4, style={'margin-top':15}),
        dbc.Col([top_models_table], width=6, style={'margin-top':15}),
    ], style={'background-color': 'rgba(211, 211, 211, 0.05)'}),
    dbc.Row([
        ts_bar_chart
    ], style={'margin-top': 4}),
    dbc.Row([
        details_table
    ], style={'background-color': 'rgba(211, 211, 211, 0.05)'})
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=10000)
