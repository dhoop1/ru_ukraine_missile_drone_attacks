import gspread
from datetime import date

gc = gspread.service_account()

intro_ref = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('Introduction')

cell_last_updated = intro_ref.find("Last Updated")
cell_mad = intro_ref.find("missile_attacks_daily")

intro_ref.update_cell(cell_mad.row, cell_last_updated.col, date.today().strftime("%m/%d/%Y"))

#print(isinstance(cell_mad.row, int), isinstance(cell_last_updated.col, int))