import gspread

gc = gspread.service_account()

mad_ref = gc.open('csis_kaggle_ru_ukraine_attacks').worksheet('Introduction')

cell_last_updated = mad_ref.find("Last Updated")
cell_mad = mad_ref.find("missile_attacks_daily")

print(cell_mad.row, cell_last_updated.col)