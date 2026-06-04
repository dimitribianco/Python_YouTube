# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:40:40 2026

@author: Dimitri Bianco

Purpose: To learn basic data manipulation.
"""

# In line comment

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:/Users/Bianco/Videos/Data/fake_data.csv')

print(data.Profit)
print(data['Profit'])

# Calculate the average of the Profit
#data['Profit'].mean()
# $39,300.00
avg_profit = data['Profit'].mean()

data['Profit'].min()
data['Profit'].max()

ford = data[data['Price'] == 'F']
goldman = data[data['Price'] == 'GS']

#This is used in examples online.
# df --> dataframe

data.dropna(inplace=True)
data = data.dropna()

data.fillna(0, inplace=True)
data['Profit'].mean()
# 15,720.00


data['Price'].replace(np.nan, 'Cool Stock', inplace=True)

#Renaming columns
data.rename(columns={'Price': 'Stock'}, inplace=True)

data.set_index(['Date', 'Stock'], inplace=True)










