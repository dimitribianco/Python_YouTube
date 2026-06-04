# -*- coding: utf-8 -*-
"""
Author: Dimitri Bianco

Purpose: Explore Data
"""

import pandas as pd


data = pd.read_csv(r'C:\Users\dimit\Documents\GitHub\Python_YouTube\Credit Data.csv')

#Let's quickly look at the data.
print(data.head(25))

#The head() function sucks as it skips columns.
#Just manually open the dataframe in Spyder.
#Color coding is nice.
#Quickly see there are some missing values.

#Let's look at missing values in each column.
    #Often in large datframes it can be hard to see missing values.
data_analysis = pd.DataFrame(data.isna().sum(), columns=['Miss_Cnt'])


#Only "Defualt Date" has missing values.
    #This makes sense. Many customers don't default.
    # Is 450 obs a lot?
#Let's calculate percentages as they are more helpful.
data_analysis['Miss_Pct'] = data_analysis['Miss_Cnt']/len(data)
    #Remeber it is decimal format!

#Get unique values (this is information available).
data_analysis['Unique_Cnt'] = data.nunique()
















