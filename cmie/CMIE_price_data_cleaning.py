#%%

import pandas as pd
from datetime import datetime
import numpy as np
import os

folder_loc = r'E:\Internship_16 Alpha Tech\CMIE data\Daily Price Data\Excel Files\Results'
file_name = 'concatenated_output.xlsx'

full_path = os.path.join(folder_loc, file_name)

#Reading the excel file
print('Reading the excel file')
data = pd.read_excel(full_path, header=[0,1], index_col=0, sheet_name='Sheet1')


print('Excel file read successfully')

#Cleaning the data
print('Cleaning the data')

#Removing the first row of the data
print('Removing the first row of the data')
data.set_index(('Date','Unnamed: 1_level_1'), inplace=True, drop=True)
print('First row removed successfully')

#Renaming the columns
print('Renaming the columns')
data = data.rename(columns={'Adjusted Closing Price': 'Close', 'Adjusted Opening Price': 'Open'}, level=1)
print('Columns renamed successfully')

#Converting the index from datetime to date format
print('Converting the index from datetime to date format')
data.index = data.index.date
data.index.name = 'Date'
print('Index converted successfully')


#Adjusting the column order
print('Adjusting the column order')

data = data.sort_index(level = 1 , axis = 1)
cols = data.columns

open_cols = [col for col in cols if 'Open' in col]
close_cols = [col for col in cols if 'Close' in col]

new_order = open_cols + close_cols

data = data[new_order]

data = data.swaplevel(axis=1)

print('Column order adjusted successfully')


#Saving the cleaned data
print('Saving the cleaned data')
data.to_csv('CMIE_Price_Data_Cleaned.csv')
print('Data saved successfully')