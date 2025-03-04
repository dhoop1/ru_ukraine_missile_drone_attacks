# %% [markdown]
# ## Access Kaggle API and upload .csv datasets into CoLab

# %%
#@title ### Access Kaggle datasets for Ukraine missile/UAV attacks

import subprocess, os

subprocess.check_call(['pip', 'install', '--upgrade pip'])
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


# %%
#@title ### Create kaggle_last_updated (date)

# Create kaggle_last_updated variable for the date that the kaggle dataset was last updated
import zipfile
import datetime
with zipfile.ZipFile("massive-missile-attacks-on-ukraine.zip", "r") as zip_ref:
    for file_info in zip_ref.infolist():
       kaggle_info_date = file_info.date_time

kaggle_last_updated = datetime.date(kaggle_info_date[0], kaggle_info_date[1], kaggle_info_date[2])
kaggle_last_updated_str = kaggle_last_updated.strftime("%m/%d/%Y")

# %%
#@title ### Early One-off Corrections

# add correct 'source' for 12/27-12/28 2024 attack
attacks_df_orig.loc[(attacks_df_orig['time_start']=="2024-12-27 23:00") & (attacks_df_orig['time_end']=="2024-12-28 08:30"), 'source'] = "kpszsu/posts/pfbid0HXuzCiwVcdjqxxtzKYADAdQjcLhjGEDhhbXTftMr6NshdikKmET3LFWJePHCCg2Cl"
attacks_df.loc[(attacks_df['time_start']=="2024-12-27 23:00") & (attacks_df['time_end']=="2024-12-28 08:30"), 'source'] = "kpszsu/posts/pfbid0HXuzCiwVcdjqxxtzKYADAdQjcLhjGEDhhbXTftMr6NshdikKmET3LFWJePHCCg2Cl"

# %% [markdown]
# ### Set-up Google Sheets Access

# %%
subprocess.check_call(['pip', 'install', 'gspread', '--quiet'])

# %%
#@title #### Authorize gspread as gc

import gspread
import os, json

# Authorize gspread with Google Cloud API (per https://docs.gspread.org/en/v6.1.3/oauth2.html#enable-api-access-for-a-project)

try:
    # gspread auth method for Render.com build -- requires Env Variable Secret
    gc = gspread.service_account(filename='/etc/secrets/GOOGLE_KAGGLE_CREDENTIALS')

except FileNotFoundError:
    if os.environ.get("GOOGLE_KAGGLE_CREDENTIALS") is None:

        # gspread auth method for VS Code desktop -- requires upload into C:/(user)~/APPDATA
        gc = gspread.service_account()

    else:
        # gspread auth method for GitHub Codespaces -- requires Secret value in GitHub 'ru_ukraine_missile_drone_attacks' repo
        credentials = json.loads(os.environ.get("GOOGLE_KAGGLE_CREDENTIALS"))
        gc = gspread.service_account_from_dict(credentials)

# %%
#@title #### Update 'pitferm_kaggle_ru_ukraine_attacks' worksheet with cleaned 'attacks_df' data

# Select worksheet from 'pitferm_kaggle_ru_ukraine_attacks' spreadsheet
mad_ref = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('missile_attacks_daily')

# Clear selected worksheet
mad_ref.clear()

# Import 'missile_attacks_daily' (from cleaned attacks_df) into gworksheet
mad_upload = attacks_df.copy()
mad_ref.update([mad_upload.columns.values.tolist()] + mad_upload.values.tolist())

# %%
#@title #### Update 'pitferm_kaggle_ru_ukraine_attacks' Introduction worksheet with kaggle_last_updated date

# Create var for Introduction worksheet
intro_ref = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('Introduction')

# Index row and column (using contents cells)
cell_last_updated = intro_ref.find("Last Updated")
cell_mad = intro_ref.find("missile_attacks_daily")

# Update cell at indexed coordinates with kaggle_last_updated
intro_ref.update_cell(cell_mad.row, cell_last_updated.col, kaggle_last_updated_str)

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
reassign = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('reassign')
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
agg = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('aggregate')
agg_df = pd.DataFrame(agg.get_all_values(), columns=agg.row_values(1))
agg_df = agg_df.iloc[1:, :-1].reset_index(inplace=False, drop=True) # remove duplicate header row and drop last column

for ia, rowa in agg_df.iterrows():
    cmodel_df.loc[cmodel_df['model_orig']==rowa.model_orig_1, 'model'] = rowa.model
    cmodel_df.loc[cmodel_df['model_orig']==rowa.model_orig_2, 'model'] = rowa.model

# %% [markdown]
# #### Clean-up: Modify v2

# %%
# Create mdfy_df from gworksheet 'modify', drop last three columns, and update dtypes
mdfy = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('modify')
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
#@title ### Clean-up: Add 'hit_rate' to cmodel_df

# Add hit_rate cmodel_df (simpler to do so here rather than update all above code components involving gspread)

cmodel_df['hit_rate'] = (cmodel_df['hit'] / cmodel_df['launched']).round(2)

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
print("1. C-400 and Iskander-M	7.0	7.0 -- correction from data capture error in Kaggle dataset")
print("2. Iskander-M/KN-23	-1.0	3.0 -- due to rounding during clean-up")
print("3. Orlan-10 and ZALA	-1.0	-1.0 -- due to rounding during clean-up")
print("4. X-101/X-555 and Kalibr and X-59/X-69 -6.0	-5.0 -- due to data correction as specified in Kaggle dataset.", "\n")

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

detail = gc.open('pitferm_kaggle_ru_ukraine_attacks').worksheet('model_detail')
detail_df = pd.DataFrame(detail.get_all_values(), columns=detail.row_values(1))
detail_df = detail_df.iloc[1:, :].reset_index(inplace=False, drop=True)

if detail_df.isna().sum().sum() == 0:
  print("Complete (no NA entries).")
else:
  print("Check upload datasheet--NA entries present.")

# %%
#@title #### Merge cmodel_df and detail_df with an inner join

# prompt: Join on inner cmodel_df and detail_df, so that each indexed row in cmodel_df has columns added that mapped over from detail_df.

# Merge cmodel_df and detail_df
merged_df = pd.merge(cmodel_df, detail_df, on='model', how='inner')

# Round 'miss_rate' to two decimal places
merged_df['miss_rate'] = merged_df['miss_rate'].round(2)

# Set 'start_date' and 'end_date' to datetime
merged_df['time_start'] = pd.to_datetime(merged_df['time_start'], format='mixed')
merged_df['time_end'] = pd.to_datetime(merged_df['time_end'], format='mixed')

# Convert unit_cost_usd with pd.to_numeric()
merged_df['unit_cost_usd'] = pd.to_numeric(merged_df['unit_cost_usd'].str.replace(',', ''), errors='coerce')

# Sort by 'time_end'
merged_df = merged_df.sort_values(by='time_end', ascending=False)

# Time-constrain merged_df for data before 28 Dec 2024
#merged_df = merged_df[merged_df['time_end'] < pd.to_datetime('2025-01-01')]

print('Merge complete.')

# %% [markdown]
# ## Dash App

# %% [markdown]
# ### DPX Components

# %% [markdown]
# #### Install and Load Dash/Plotly modules-functions

# %%
#@title Install dash/plotly modules

subprocess.check_call(['pip', 'install', 'dash', 'plotly', 'dash-html-components', 'dash-core-components', 'dash-bootstrap-components', '--quiet'])

# %%
#@title Load most Dash/Plotly functions

import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# %% [markdown]
# #### DPX app: details_table

# %%
#@title DPX app: details_table

# prompt: Using Dash Plotly and from merge_df, create a table that groups merged_df by the columns from detail_df (i.e. detail_df.columns) and aggregates 'launched', 'hit', 'miss', 'destroyed', and 'miss_rate' the rows.

import dash
from dash import dash_table, State
from dash.dash_table.Format import Format, Group, Scheme

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

# Create the data for the PX table, using df.groupby().agg()
details_data = merged_df.groupby(detail_df.columns.tolist())[['launched', 'hit', 'miss', 'destroyed']].agg({ # 'miss_rate', 'destroyed_rate'
            'launched': 'sum',
            'hit': 'sum',
            'miss': 'sum',
            'destroyed': 'sum',
            #'miss_rate': 'mean',
            #'destroyed_rate': 'mean'
        }).reset_index()

# Insert 'miss_rate' and 'destroyed_rate' (rounded to 2 float) into details_data
details_data['miss_rate'] = (details_data['miss'] / details_data['launched']).round(2)
details_data['destroyed_rate'] = (details_data['destroyed'] / details_data['launched']).round(2)

# Sort the details_data into category and type, descending by max 'launched' value per category and type, then by 'launched' within each 'category-type'
details_data['max_l_cat'] = details_data.groupby(['category'])['launched'].transform('max')
details_data['max_l_type'] = details_data.groupby(['type'])['launched'].transform('max')
details_data = details_data.sort_values(by=['max_l_cat', 'max_l_type', 'launched'], ascending=False, inplace=False)
details_data.drop(columns=['max_l_cat', 'max_l_type'], inplace=True) # drop temporary columns used for sorting

# Create dict of column names to use in detail_table
details_columns = [{"name": i, "id": i} for i in detail_df.columns.tolist() + ['launched', 'hit', 'miss', 'destroyed', 'miss_rate', 'destroyed_rate']]

text_cols = detail_df.columns.drop('unit_cost_usd').to_list() # list of columns to format as text
thousands_cols = ['unit_cost_usd', 'launched', 'hit', 'miss', 'destroyed'] # list of columns to format with thousands comma
percent_cols = ['miss_rate', 'destroyed_rate'] # list of columns to format as percent

for h, i in enumerate(details_columns): # conditionally append 'type' and 'format' to details_columns dict
    if i.get('name') in text_cols:
        details_columns[h]['type'] = 'text'
    if i.get('name') in thousands_cols:
        details_columns[h]['type'] = 'numeric'
        details_columns[h]['format'] = format=Format().group(True)
    if i.get('name') in percent_cols:
        details_columns[h]['type'] = 'numeric'
        details_columns[h]['format'] = Format(precision=0, scheme=Scheme.percentage)

# PX table data and html formatting
details_table = html.Div([
    html.H2("Missile & Drone Model Details"),
    dbc.Button("Reset Sorting", id="reset-sorting-btn", color="primary", outline=True, size="sm"),
    dash_table.DataTable(
        id='details-table',
        columns=details_columns,
        data=details_data.to_dict('records'),
        style_table={'margin-top': '10px'},
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
        style_cell={'fontSize': 8, 'minwidth': '4%'},
        style_cell_conditional=[
            {'if': {'column_id': ['model', 'type', 'detail', 'maneuverability']}, 'width': '7.75%'}, #31%
            {'if': {'column_id': ['category', 'range', 'altitude', 'speed']}, 'width': '6.5%'}, #26%
            {'if': {'column_id': ['payload_kg', 'accuracy', 'unit_cost_usd']}, 'width': '5%'}, #15%
            {'if': {'column_id': ['launched', 'hit', 'miss', 'destroyed', 'miss_rate']}, 'width': '4.5%'}, #22.5%
            {'if': {'column_id': ['destroyed_rate']}, 'width': '5.5%'}, #5.5%
        ],
        #style_table={'overflowX': 'auto'},
        sort_action='native',
        sort_mode='multi',
        page_size=10,
        #style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
    )
], style={'margin-top': '5px'})

# %% [markdown]
# #### DPX app: attacks_table (data)

# %%
# Create initial attacks_table_data with group-summed cols
attacks_table_df = merged_df.copy()
attacks_table_df['total cost'] = (attacks_table_df['launched'] * attacks_table_df['unit_cost_usd'])
attacks_table_data = attacks_table_df.groupby('category', observed=False)[['launched', 'hit', 'miss', 'destroyed', 'total cost']].sum().sort_values('launched', ascending=False).reset_index()
attacks_table_data['total cost'] = attacks_table_data['total cost'].round(0)

# Add rate cols to attacks_table_data
attacks_table_data['hit_rate'] = (attacks_table_data['hit'] / attacks_table_data['launched']).round(2)
attacks_table_data['miss_rate'] = (attacks_table_data['miss'] / attacks_table_data['launched']).round(2)
attacks_table_data['destroyed_rate'] = (attacks_table_data['destroyed'] / attacks_table_data['launched']).round(2)

# Add avg_unit_cost col to attacks_table_data
avg_unit_cost_catg = (attacks_table_df.groupby('category')['total cost'].sum() / attacks_table_df.groupby('category')['launched'].sum()).round(-3)
attacks_table_data['avg unit cost'] = attacks_table_data['category'].map(avg_unit_cost_catg)

# Create separate datasets for main categories (attacks_table_data_main) and incomplete categories (attacks_table_data_minor)
attacks_table_data_main = attacks_table_data[~attacks_table_data['category'].isin(['UAV - recon', 'aerial bomb', 'IRBM'])].copy()
attacks_table_data_minor = attacks_table_data[attacks_table_data['category'].isin(['UAV - recon', 'aerial bomb', 'IRBM'])].copy().reset_index(drop=True)

# Create initial totals row with group-summed cols, and insert into attacks_table_data_main
attacks_table_df_main = attacks_table_df[~attacks_table_df['category'].isin(['UAV - recon', 'aerial bomb', 'IRBM'])].copy().reset_index(drop=True)
attacks_table_totals_main = attacks_table_df_main.groupby('category', observed=False)[['launched', 'hit', 'miss', 'destroyed', 'total cost']].sum().sum()
#attacks_table_totals_main['total cost'] = attacks_table_totals_main['total cost'].round(0)
attacks_table_data_main.loc['Total', ('launched', 'hit', 'miss', 'destroyed', 'total cost')] = attacks_table_totals_main.values

# Insert total averages into attacks_table_data, specify 'Total' (for 'category' column), and reset index
attacks_table_data_main.loc['Total', 'hit_rate'] = (attacks_table_df_main['hit'].sum() / attacks_table_df_main['launched'].sum()).round(2)
attacks_table_data_main.loc['Total', 'miss_rate'] = (attacks_table_df_main['miss'].sum() / attacks_table_df_main['launched'].sum()).round(2)
attacks_table_data_main.loc['Total', 'destroyed_rate'] = (attacks_table_df_main['destroyed'].sum() / attacks_table_df_main['launched'].sum()).round(2)
attacks_table_data_main.loc['Total', 'avg unit cost'] = (attacks_table_df_main['total cost'].sum() / attacks_table_df_main['launched'].sum()).round(-3)
attacks_table_data_main.loc['Total', 'category'] = 'Total'
attacks_table_data_main = attacks_table_data_main.reset_index(drop=True)


# %%
#@title attacks_table (html)

# prompt: Using Dash Plotly and from merged_df, create a table which displays the 'launched', 'hit', 'miss', and 'destroyed' summed-up counts, grouped by 'category'. Title this table as, "Missile Attacks by Category, for All Time".

from dash import dash_table, html
from dash import dash_table
from dash.dash_table.Format import Format, Group, Scheme


# Define cell style for 'Table' row
total_style = [
    {
        'if': {
            'filter_query': '{category} = "Total"',  # Filter for rows where category is 'Total'
            'column_id': list(attacks_table_data.columns)  # Apply across all columns
        },
        'backgroundColor': 'rgba(211, 211, 211, 0.1)',  # Set background color
    }
]


# Define attacks_table via html
attacks_table = html.Div([
    html.H4("Breakdown by Category"),
    dash_table.DataTable(
        id='attacks-table',
        #columns=[{"name": i, "id": i} for i in ['category', 'launched', 'hit', 'miss', 'destroyed']],
        columns=[
            {"name": 'Category', "id": 'category', "type": 'text'},  # Set 'category' to string/object type
            {"name": 'Launched', "id": 'launched', "type": 'numeric', "format": Format().group(True)},  # Set 'launched' to numeric
            #{"name": 'hit', "id": 'hit', "type": 'numeric', "format": Format().group(True)},  # Set 'hit' to numeric
            {"name": 'Hit', "id": 'hit_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            #{"name": 'miss', "id": 'miss', "type": 'numeric', "format": Format().group(True)},  # Set 'miss' to numeric
            {"name": 'Miss', "id": 'miss_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            #{"name": 'destroyed', "id": 'destroyed', "type": 'numeric', "format": Format().group(True)},  # Set 'destroyed' to numeric
            {"name": 'Destroyed', "id": 'destroyed_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            {"name": 'Avg Unit Cost', "id": 'avg unit cost', "type": 'numeric', "format": Format().group(True)},  # Set 'unit_cost_usd' to numeric
            {"name": 'Total Cost', "id": 'total cost', "type": 'numeric', "format": Format().group(True)},  # Set 'cost (US$M)' to numeric
        ],
        data=attacks_table_data_main.to_dict('records'),
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
            'margin-bottom': '2px',
        },
        style_cell={'fontSize': 14, 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'category'}, 'width': '20%'}, # Set width for 'category' column
            #{'if': {'column_id': ['launched', 'hit', 'hit_rate' 'miss', 'miss_rate', 'destroyed', 'destroyed_rate']}, 'width': '11%', 'format': {'specifier': ',.0f'}}  # Equal width for other columns, and center-aligned text
            {'if': {'column_id': ['launched', 'hit_rate' 'miss_rate', 'destroyed_rate']}, 'width': '13%', 'format': {'specifier': ',.0f'}},
            {'if': {'column_id': ['avg unit cost', 'total cost']}, 'width': '15%', 'format': {'specifier': ',.0f'}},
        ],
        style_data_conditional=total_style  # apply style for 'Total' row
    ),
])

# %%
#@title attacks_table_minor (html)

attacks_table_minor = html.Div([
    html.H6("Categories with Incomplete Data", style={'margin-top': '6px'}),
    dash_table.DataTable(
        id='attacks-table-minor',
        #columns=[{"name": i, "id": i} for i in ['category', 'launched', 'hit', 'miss', 'destroyed']],
        columns=[
            {"name": 'Category', "id": 'category', "type": 'text'},  # Set 'category' to string/object type
            {"name": 'Launched', "id": 'launched', "type": 'numeric', "format": Format().group(True)},  # Set 'launched' to numeric
            #{"name": 'hit', "id": 'hit', "type": 'numeric', "format": Format().group(True)},  # Set 'hit' to numeric
            {"name": 'Hit', "id": 'hit_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            #{"name": 'miss', "id": 'miss', "type": 'numeric', "format": Format().group(True)},  # Set 'miss' to numeric
            {"name": 'Miss', "id": 'miss_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            #{"name": 'destroyed', "id": 'destroyed', "type": 'numeric', "format": Format().group(True)},  # Set 'destroyed' to numeric
            {"name": 'Destroyed', "id": 'destroyed_rate', "type": 'numeric', "format": Format(precision=0, scheme=Scheme.percentage)},
            {"name": 'Avg Unit Cost', "id": 'avg unit cost', "type": 'numeric', "format": Format().group(True)},  # Set 'unit_cost_usd' to numeric
            {"name": 'Total Cost', "id": 'total cost', "type": 'numeric', "format": Format().group(True)},  # Set 'cost (US$M)' to numeric
        ],
        data=attacks_table_data_minor.to_dict('records'),
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
            'margin-right': '20px',
            'margin-bottom': '2px',
        },
        style_cell={'fontSize': 12, 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'category'}, 'width': '20%'}, # Set width for 'category' column
            #{'if': {'column_id': ['launched', 'hit', 'hit_rate' 'miss', 'miss_rate', 'destroyed', 'destroyed_rate']}, 'width': '11%', 'format': {'specifier': ',.0f'}}  # Equal width for other columns, and center-aligned text
            {'if': {'column_id': ['launched', 'hit_rate' 'miss_rate', 'destroyed_rate']}, 'width': '13%', 'format': {'specifier': ',.0f'}},
            {'if': {'column_id': ['avg unit cost', 'total cost']}, 'width': '15%', 'format': {'specifier': ',.0f'}},
        ],
        style_data_conditional=total_style  # apply style for 'Total' row
    ),
    html.Div([
        html.Em("Note: 'aerial bomb' and 'UAV - recon' are significantly under-represented in the Kaggle dataset, and 'IRBM' is miniscule.")
    ], className="card-text", style={'fontSize': '11px', 'textAlign': 'left', 'margin-left': '2px', 'margin-bottom': '10px'})
])

# %% [markdown]
# #### DPX app: intro_info

# %%
#@title intro_info

# brief description and image

# Create intro 'Description' as dbc.Card()
intro_text = dbc.Card(
    dbc.CardBody([
        html.H4("Description", className="card-title", style={'textAlign': 'center'}),
        html.P([
            "This ",
            html.A("dashboard", href="https://ru-ukraine-missile-drone-attacks.onrender.com/"),
            " presents metrics on Russian missiles & drones launched against Ukraine (Oct 2022-present).",
            html.Div(html.Img(src="assets/ukraine_map_flag_transparent_cropped.png", alt="ukraine-map", style={'width': '90%'}), style={'textAlign': 'center'}),
            "The data source is Kaggle-hosted ",
            html.A("Massive Missile Attacks on Ukraine", href="https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine", target="_blank"),
            ", which weekly sources its data from the Ukrainian Air Force social accounts (e.g. ",
            html.A("KpsZSU", href="https://facebook.com/kpszsu"),
            ", ",
            html.A("PvKPivden", href="https://facebook.com/PvKPivden"),
            ") and is also featured by ",
            html.A("CSIS", href="https://www.csis.org/programs/futures-lab/projects/russian-firepower-strike-tracker-analyzing-missile-attacks-ukraine"),
            ". Hooper Consulting ",
            html.A("cleans and enhances", href="https://docs.google.com/spreadsheets/d/1Zs705hRN7HfUOOhTZN2nNIPB6SAeKaxU1AQAkGZinzk/edit?usp=sharing"),
            " the Kaggle data, and delivers this dashboard via a Python Dash app, ",
            html.A("GitHub", href="https://github.com/dhoop1/ru_ukraine_missile_drone_attacks"),
            ", and Render.",
        ], className="card-text", style={'fontSize': '11.5px', 'textAlign': 'center'},
        ),
        html.P([
            #html.Em("Kaggle data updated: " + "01/05/2025"),
            html.Em("Kaggle data updated: " + kaggle_last_updated_str),
        ], className="card-text", style={'fontSize': '11.5px', 'textAlign': 'center', 'margin': '0px'}
        )
    ]),
    #style={'border': '1px solid lightgray', "margin-bottom":"15px"},
    style={'border': 'none', 'textAlign': 'center', 'background-color': 'rgba(211, 211, 211, 0.07)'} # set faint gray background
)

# %% [markdown]
# #### DPX app: metrics_bar

# %%
#@title metrics_bar (data)

# merged_df_main
merged_df_main = merged_df[~merged_df['category'].isin(['UAV - recon', 'aerial bomb', 'IRBM'])].copy().reset_index(drop=True)

# days_at_war
from datetime import date
#min_date = pd.to_datetime(merged_df['time_start'], format="mixed").min()
start_date = pd.to_datetime("02/24/2022 12:00:00") # First day of Russia invasion against Ukraine (24 Feb 2022), as reported by the Ukrainian Air Force
today = date.today()
days_at_war = (today - start_date.date()).days

# weeks_count
min_date = pd.to_datetime(merged_df_main['time_start'], format="mixed").min()
today = date.today()
last_day_2024 = pd.to_datetime("12/31/2024")
weeks_count = round((today - min_date.date()).days / 7, 0)

# total_launched
total_launched = merged_df_main['launched'].sum()

# total_ru_mad_cost
#merged_df['unit_cost_usd'] = merged_df['unit_cost_usd'].astype(str).str.replace(",", "", regex=False)
#merged_df['unit_cost_usd'] = pd.to_numeric(merged_df['unit_cost_usd'], errors='coerce')
attack_cost = merged_df_main['launched'] * merged_df_main['unit_cost_usd'].astype(float)
total_ru_mad_cost = (attack_cost.sum()/1000000).round(0)

# weekly_avg_launched
weekly_avg_launched = (total_launched / weeks_count).round(0)

# weekly_avg_cost
weekly_avg_cost = (total_ru_mad_cost / weeks_count).round(0)

# last_30days_launched
merged_df_main['time_end'] = pd.to_datetime(merged_df['time_end'], format="mixed")
#last_30days = merged_df_main[merged_df_main['time_end'] >= (last_day_2024 - pd.Timedelta(days=30))]
last_30days = merged_df[merged_df['time_end'] >= (pd.Timestamp(today) - pd.Timedelta(days=30))]
last_30days_launched = last_30days['launched'].sum()

# last_30days_cost
last_30days_cost = ((last_30days['launched'] * last_30days['unit_cost_usd'].astype(float)).sum() / 1000000).round(0)

# %%
#@title v2 metrics_bar (html.Div)

# prompt: Create metrics_bar2 = html.Div() which contains days_at_war, total_launched, total_ru_mad_cost, and last_30days_launched as four Cards horizontally adjacent (which "share" a common background and outline), with the descriptor in html.H6() and the variable in html.P() smaller font size.

metrics_data = [{
        #'days-at-war': days_at_war,  # Reference actual variable
        'total-launched': total_launched,  # Reference actual variable
        'total-cost': total_ru_mad_cost,  # Reference actual variable
        'weekly-avg-launched': weekly_avg_launched,  # Reference actual variable
        'weekly-avg-cost': weekly_avg_cost,  # Reference actual variable
        'last-30days-launched': last_30days_launched,  # Reference actual variable
        'last-30days-cost': last_30days_cost,  # Reference actual variable
    }]

metrics_bar4 = html.Div([
    html.H4("Summary Metrics (Oct 2022-Present)"), # style={'textAlign': "center"}
    dbc.Row([
        dash_table.DataTable(
            id='metrics-table',
            columns=[
                #dict(id='days-at-war', name="Days \n at War", type='numeric', format=Format().group(True)),
                dict(id='total-launched', name="Total \n Launched", type='numeric', format=Format().group(True)),
                dict(id='total-cost', name="Total Cost ($M)", type='numeric', format=Format().group(True)),
                dict(id='weekly-avg-launched', name="Weekly Avg \n Launched", type='numeric', format=Format().group(True)),
                dict(id='weekly-avg-cost', name="Weekly Avg \n Cost ($M)", type='numeric', format=Format().group(True)),
                dict(id='last-30days-launched', name="Last 30 Days Launched", type='numeric', format=Format().group(True)),
                dict(id='last-30days-cost', name="Last 30 Days \n Cost ($M)", type='numeric', format=Format().group(True)),
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
                'margin': 3,
                'padding': '2% 2% 2% 0%'
            },
            style_data={
                'backgroundColor': 'transparent',
                'color': 'lightgray',
                'border': 'none',
            },
            style_cell={
                'fontSize': 14,
                'textAlign': 'center',
                'width': '16.7%',
                'border': 'none',
            },
            style_header_conditional=[
                {'if': {'column_id': ['total-launched', 'weekly-avg-launched', 'last-30days-launched']}, 'background-color': 'rgba(211, 211, 211, 0.2)'},
                {'if': {'column_id': ['total-cost', 'weekly-avg-cost', 'last-30days-cost']}, 'background-color': 'rgba(211, 211, 211, 0.1)'}
            ],
            style_data_conditional=[
                {'if': {'column_id': ['total-launched', 'weekly-avg-launched', 'last-30days-launched']}, 'background-color': 'rgba(211, 211, 211, 0.1)'},
                #{'if': {'column_id': ['days-at-war', 'total-launched']}, 'background-color': 'rgba(211, 211, 211, 0.05)'}
            ],
        )
    ]) # style={'border': '1px solid lightgray', 'margin': 4, 'padding': '2%'}
])

# %% [markdown]
# #### DPX app: attack_size

# %%
#@title attack_size_avg_data

# Create mod_category list (as a list of 'category' values to include)
as_category_list = ['UAV - munition', 'cruise missile', 'ballistic missile', 'anti-air missile']

# Create attack_size_df, convert time_end datetime to date, and filter to columns in mod_category_list
attack_size_df = merged_df.copy()
attack_size_df['time_end'] = pd.to_datetime(attack_size_df['time_end'], format="mixed").dt.date
attack_size_df = attack_size_df.loc[attack_size_df['category'].isin(as_category_list), :]

# Create bins/labels for daily 'attack size' counts
bins = [0, 10, 50, 100, float('inf')]
labels = ['<10', '10-49', '50-99', '100+']

# Create attack_size_list (as a header-labeled list of 'attack size' labels)
attack_size_list = ['attack size'] + labels

# Create hit_by_category (as pivot table of 'launched' and 'hit' summed by 'end_date' and 'category')
launched_by_category = attack_size_df.pivot_table(index='time_end', columns='category', values=('launched', 'hit'), aggfunc='sum', fill_value=0)
#launched_by_category = launched_by_category.drop(columns=[('launched', 'UAV - recon'), ('launched', 'aerial bomb'), ('launched', 'IRBM'), ('hit', 'UAV - recon'), ('hit', 'aerial bomb'), ('hit', 'IRBM')])
launched_by_category['Avg Hits'] = launched_by_category['hit'].sum(axis=1) # titling as 'Avg Hits' now, and will later apply .mean()
launched_by_category['Avg Launched'] = launched_by_category['launched'].sum(axis=1) # titling as 'Avg Launched' now, and will later apply .mean()
launched_by_category['attack size'] = pd.cut(launched_by_category['Avg Launched'], bins=bins, labels=labels, right=False) # adds new 'attack size' column per bins/labels

# Create daily_avg_hit_by_category (with grouped totals for 'Avg Launched', 'Avg Hits', and '# attacks')
daily_avg_launched_by_category = launched_by_category.groupby('attack size', observed=False)['Avg Hits'].mean().reset_index() # now becomes a .mean()
daily_avg_launched_by_category.columns = ['attack size', 'Avg Hits']
daily_avg_launched_by_category['Avg Launched'] = launched_by_category.groupby('attack size', observed=False)['Avg Launched'].mean().reset_index()['Avg Launched'] # now becomes a .mean()
daily_avg_launched_by_category['# attacks'] = launched_by_category.groupby('attack size', observed=False)['Avg Launched'].count().values # counts of 'attack size' labels
daily_avg_launched_by_category = daily_avg_launched_by_category.round(1)

# Create launched_means (by group-averaging from hit_by_category, transposing, and concatting 'Avg Launched')
launched_means2 = launched_by_category.groupby(['attack size'], observed=False)[[('launched', 'UAV - munition'), ('launched', 'cruise missile'), ('launched', 'ballistic missile'), ('launched', 'anti-air missile')]].mean().reset_index()
launched_means2 = launched_means2['launched']
launched_means2.columns = ["(L) " + c for c in as_category_list] # differentiate 'category' labels for 'launched' (in same df/table)
launched_means2 = launched_means2.T.round(1)
launched_means2 = pd.concat([daily_avg_launched_by_category['Avg Launched'].to_frame(name='Avg Launched').T, launched_means2], ignore_index=False).reset_index()
launched_means2.columns = attack_size_list

# create hit_means
hit_means2 = launched_by_category.groupby(['attack size'], observed=False)[[('hit', 'UAV - munition'), ('hit', 'cruise missile'), ('hit', 'ballistic missile'), ('hit', 'anti-air missile')]].mean().reset_index()
hit_means2 = hit_means2['hit']
hit_means2.columns = ["(H) " + c for c in as_category_list] # differentiate 'category' labels for 'hit' (in same df/table)
hit_means2 = hit_means2.T.round(1)
hit_means2 = pd.concat([daily_avg_launched_by_category['Avg Hits'].to_frame(name='Avg Hits').T, hit_means2], ignore_index=False).reset_index()
hit_means2.columns = attack_size_list

# Create attack_size_avg_data (by concatenating 'attack size', '# attacks', launched_means2, and hit_means2)
attack_size_avg_data = pd.concat([daily_avg_launched_by_category.set_index('attack size')['# attacks'].to_frame().T.reset_index().rename(columns={'index': 'attack size'}), launched_means2, hit_means2], ignore_index=True)

# %%
#@title attack_size_count (html)

from dash import dash_table, html
from dash import dash_table
from dash.dash_table.Format import Format, Group, Scheme

avg_launched_style = {
    'if': {
        'filter_query': '{attack size} = "Avg Launched"',  # Filter for rows where category is '# attacks'
        'column_id': attack_size_list  # Apply across all columns
    },
    'type': 'numeric', 'format': Format(precision=0, scheme=Scheme.fixed),  # Set float point
    'backgroundColor': 'rgba(211, 211, 211, 0.05)',  # Set background color
    'fontSize': '14px'
}

avg_hits_style = {
    'if': {
        'filter_query': '{attack size} = "Avg Hits"',  # Filter for rows where category is '# attacks'
        'column_id': attack_size_list  # Apply across all columns
    },
    'type': 'numeric', 'format': Format(precision=0, scheme=Scheme.fixed),  # Set float point
    'backgroundColor': 'rgba(211, 211, 211, 0.05)',  # Set background color
    'fontSize': '14px',
}

num_attacks_style = {
    'if': {
        'filter_query': '{attack size} = "# attacks"',  # Filter for rows where category is '# attacks'
        'column_id': attack_size_list  # Apply across all columns
    },
    'type': 'numeric', 'format': Format(precision=0, scheme=Scheme.fixed),
    'fontSize': '14px',
}


# Define attack_size_avg DataTable
attack_size_avg = html.Div([
    html.H4("Avg Launched and Hits, by Attack Size", style={'margin-bottom': '10px'}),
    dash_table.DataTable(
        id='attack-size-avg',
        columns=[{'id': c, 'name': c} for c in attack_size_list],
        data=attack_size_avg_data.to_dict('records'),
        #data=daily_avg_hit_by_category.set_index('attack size')['# attacks'].to_frame().T.reset_index().rename(columns={'index': 'attack size'}).to_dict('records'),
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'lightgray',
        },
        style_table={
            'overflowX': 'auto',
            'width': '100%',
            'border-top': '0',
        },
        style_cell={'fontSize': 12, 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'attack size'}, 'backgroundColor': 'rgb(30, 30, 30)', 'width': '33%'},
            {'if': {'column_id': attack_size_list[1:]}, 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
        ],
        style_data_conditional=[
            num_attacks_style,
            avg_launched_style,
            avg_hits_style
        ],
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'lightgray',
            'fontSize': '14px',
        },
    )
])

# %%
#@title attack_size_metrics (data)

# attack_size_total
attack_size_total = daily_avg_launched_by_category['# attacks'].sum()

# attacks_per_week
attacks_per_week = round(attack_size_total / weeks_count, 1)

# avg_launched_per_attack
avg_launched_per_attack = total_launched / attack_size_total

# %%
#@title attack_size_metrics (html)

attack_size_data = [{
        'attack-size-total': attack_size_total,
        'attacks-per-week': attacks_per_week,
        'avg-launched-per-attack': avg_launched_per_attack,
    }]

attack_size_bar = html.Div([
    #html.H4("Attack Size Metrics"), # style={'textAlign': "center"} -- no title currently
    dbc.Row([
        dash_table.DataTable(
            id='attack-size-bar',
            columns=[
                dict(id='attack-size-total', name="Total # Attacks", type='numeric', format=Format().group(True)),
                dict(id='attacks-per-week', name="Avg # Attacks \n per Week", type='numeric', format=Format(precision=1, scheme=Scheme.fixed)),
                dict(id='avg-launched-per-attack', name="Avg Launched \n per Attack", type='numeric', format=Format(precision=1, scheme=Scheme.fixed)),
            ],
            data=attack_size_data,
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
                'width': '16.7%',
                'border': 'none',
            },
            style_header_conditional=[
                {'if': {'column_id': ['attack-size-total', 'avg-launched-per-attack']}, 'background-color': 'rgba(211, 211, 211, 0.2)'},
                {'if': {'column_id': ['attacks-per-week']}, 'background-color': 'rgba(211, 211, 211, 0.1)'}
            ],
            style_data_conditional=[
                {'if': {'column_id': ['attack-size-total', 'avg-launched-per-attack']}, 'background-color': 'rgba(211, 211, 211, 0.1)'},
            ],
        )
    ]) # style={'margin-top': '10px'}
])

# %% [markdown]
# #### DPX app: ts_bar_chart

# %%
#@title #### dropdown_items (html)

# range_dropdown
range_dropdown = dbc.Col([
    dbc.Row(html.Label("Range:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
    dcc.Dropdown(
        id='range-dropdown',
        options=[{'label': val, 'value': val} for val in merged_df['range'].unique()],
        value=merged_df['range'].unique().tolist(),
        multi=True,
        style={'background': '#3a3f44'},
        #style={'display': 'block'}
    )
])

# speed_dropdown
speed_dropdown = dbc.Col([
    dbc.Row(html.Label("Speed:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
    dcc.Dropdown(
        id='speed-dropdown',
        options=[{'label': val, 'value': val} for val in merged_df['speed'].unique()],
        value=merged_df['speed'].unique().tolist(),
        multi=True,
        style={'background': '#3a3f44'},
        #style={'display': 'block'}
    )
])

# altitude_dropdown
altitude_dropdown = dbc.Col([
    dbc.Row(html.Label("Altitude:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-bottom':1})),
    dcc.Dropdown(
        id='altitude-dropdown',
        options=[{'label': val, 'value': val} for val in merged_df['altitude'].unique()],
        value=merged_df['altitude'].unique().tolist(),
        multi=True,
        style={'background': '#3a3f44'},
        #style={'display': 'block'}
    )
])

# %%
#@title #### ts_bar_chart

# prompt: From merged_df, create a Dash Plotly time series visualization with a stacked bar chart, and the data determined by the following radio button selections. The x-axis will be time as either week, month, quarter, or year selected by a radio button (default is month). The y-axis will be 'launched', 'hit', 'miss', or 'destroyed' as selected by a radio button (default is 'launched'). The stacks in the bar chart will be either 'category' or 'model' as selected by a radio button (default is 'category').

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

ts_bar_chart = html.Div([

    # title
    dbc.Row([html.H2("Time Series for Missile and Drone Attacks Against Ukraine")]),

    # first row - radio buttons
    dbc.Row([

        # time_period
        dbc.Col([
            dbc.Row(html.Label("Time Period:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin':3})),
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
                value='M',  # Default value
            ),
        ], style={'display': 'block', 'textAlign': 'center', 'padding-top': '5px'}),

        # yaxis
        dbc.Col([
            dbc.Row(html.Label("Y-axis:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin':3})),
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
                value='launched', # Default value
            ),
        ], style={'display': 'block', 'textAlign': 'center', 'padding-top': '5px'}),

        # stacked_bar_feature
        dbc.Col([
            dbc.Row(html.Label("Stacked Bar:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin':3})),
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
                value='category', # Default value
            ),
        ], style={'display': 'block', 'textAlign': 'center', 'padding-top': '5px'}),

        # yaxis2
        dbc.Col([
            dbc.Row(html.Label("Second Y-Axis:", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin':3})),
            dbc.Checklist(
                id='y-axis2-toggle',
                options=[{'label': 'Launched', 'value': 'launched'}],
                value=[],
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
            ),
        ], style={'display': 'block', 'textAlign': 'center', 'padding-top': '5px'}),

        # 'category' multiple-select buttons
        dbc.Col([
            html.H6("Category:", style={'fontWeight': 'bold', 'display': 'block', 'textAlign': 'center', 'margin':5}), # 'padding-top': '5px'
            dbc.Checklist(
                id='category-slice',
                options=[{'label': category, 'value': category} for category in merged_df['category'].unique()],
                value=merged_df[~merged_df['category'].str.lower().str.contains('recon')]['category'].unique().tolist(),  # Default: all categories but 'UAV - Recon' selected
                inline=True,  # Arrange options horizontally
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-secondary",
                labelCheckedClassName="active"
            ),
        ], style={'padding-top': '5px', 'textAlign': 'center'}),

    ], style={'display': 'flex', 'justify-content': 'center'}),

    # third row - ts_bar_chart
    dbc.Row([dcc.Graph(id='time-series-chart')]),


    # fourth row - details dropdown checklists (toggle display off/on)
    dbc.Row([
        #dbc.Col(type_dropdown),
        dbc.Col(range_dropdown),
        dbc.Col(speed_dropdown),
        dbc.Col(altitude_dropdown),
    ], style={'margin-bottom': '30px', 'margin-left': '10px', 'margin-right': '10px'}, justify="center")

], style={'margin-top': '5px'})

# %% [markdown]
# ### DPX @app.callback(), app.layout(), and app.server -- combined

# %%
# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

#app.config['suppress_callback_exceptions'] = True # allow app.callbacks() generated by other callbacks
app.config.suppress_callback_exceptions = True

# ---------------------------------------------------------------------------

# ts_bar_chart
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('time-period', 'value'),
    Input('y-axis-value', 'value'),
    Input('stacked-bar-feature', 'value'),
    Input('y-axis2-toggle', 'value'),
    Input('category-slice', 'value'),
    #Input('details-toggle', 'value'),
    #Input('type-dropdown', 'value'),
    Input('range-dropdown', 'value'),
    Input('speed-dropdown', 'value'),
    Input('altitude-dropdown', 'value'),
)
def update_chart(time_period, y_axis_value, stacked_bar_feature, y_axis2, category_slice, selected_range, selected_speed, selected_altitude):
    # Create a temp_df from merged_df
    ts_bar_df = merged_df.copy()

    # If toggled-on details_dropdowns, then apply filters
    #if '(show below)' in details_dropdown:

    # Filter the data based on model details 'type', 'range', 'altitude', and 'speed'
    #if selected_type:
    #    ts_bar_df = ts_bar_df[ts_bar_df['type'].isin(selected_type)]
    if selected_range:
        ts_bar_df = ts_bar_df[ts_bar_df['range'].isin(selected_range)]
    if selected_speed:
        ts_bar_df = ts_bar_df[ts_bar_df['speed'].isin(selected_speed)]
    if selected_altitude:
        ts_bar_df = ts_bar_df[ts_bar_df['altitude'].isin(selected_altitude)]

    # Add 'time_period' to ts_bar_df (by resampling 'time_end')
    ts_bar_df['time_end'] = pd.to_datetime(merged_df['time_end'], format="mixed")
    ts_bar_df['time_period'] = ts_bar_df['time_end'].dt.to_period(time_period)
    ts_bar_df['time_period'] = ts_bar_df['time_period'].astype(str)

    # 1. Generate complete time range
    all_periods = pd.period_range(
        start=ts_bar_df['time_end'].min(),
        end=ts_bar_df['time_end'].max(),
        freq=time_period
    ).astype(str).tolist()

    # Slice data (by category_slice) and groupby/sum (by stacked_bar_feature)
    sliced_data = ts_bar_df[ts_bar_df['category'].isin(category_slice)].reset_index()
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

        launched_data = ts_bar_df.groupby('time_period')['launched'].sum().reset_index()
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

    # Space legend further down on y-axis if Stacked Bar Feature is 'model'
    if stacked_bar_feature == 'model':
        fig.update_layout(
            legend=dict(
                y=-0.4
            )
        )

    # 6. Update details-dropdown


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

# app.server
server = app.server

# app.layout
app.layout = html.Div([
    html.H1(ru_ua_title),
    dbc.Row([
        dbc.Col([intro_text], width=2),
        dbc.Col([
            dbc.Row([metrics_bar4], className='pt-0 pb-6 pl-2 pr-2'),
            dbc.Row([attacks_table], className='pt-6'),
            dbc.Row([attacks_table_minor], className='pt-6 pr-10')
            ], width=6, style={'margin-top':15}),
        dbc.Col([
            dbc.Row([attack_size_avg], className='pt-6 pb-2 pl-2 pr-2'),
            dbc.Row([attack_size_bar], className='pb-3'),
            ], width=4, style={'margin-top':15}),
    ], style={'background-color': 'rgba(211, 211, 211, 0.05)'}),
    dbc.Row([
        ts_bar_chart
    ], style={'margin-top': 4}),
    dbc.Row([
        details_table
    ], style={'background-color': 'rgba(211, 211, 211, 0.05)'})

])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)