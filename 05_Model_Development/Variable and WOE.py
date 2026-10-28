# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:04:03 2026

@author: Bianco

Purpose: Explore Data
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.read_csv(r'C:\Users\Bianco\Documents\GitHub\Python_YouTube\Credit Data_v3.csv')

#Let's quickly look at the data.
print(data.head(25))

data_sm = data.head(25)

#The head() function sucks as it skips columns.
#Just manually open the dataframe in Spyder.
#Color coding is nice.
#Quickly look for missing data or other issues.

#Let's look at missing values in each column.
    #Often in large dataframes it can be hard to see missing values.
data_analysis = pd.DataFrame(data.isna().sum(), columns=['Miss_Cnt'])

#Only "Default Date" has missing values.
    #This makes sense. Many customers don't default.
    #Is 450 obs a lot?
#Let's calculate percentages as they are more helpful.
data_analysis['Miss_Pct'] = data_analysis['Miss_Cnt']/len(data)
    #Remember it is decimal format!

#Get unique values (this is information available).
data_analysis['Unique_Cnt'] = data.nunique()


#6/4/26
#Create the dependent variable
data['Default'] = np.where(data[['Default Date']].isnull(), 0, 1)

#Drop non-numerica values/columns like dates
num_data = data.drop(columns=['Default Date', 'Contract Date'])

#Correlations
correlations = num_data.corr(method='pearson')

#Correlation Heat Map
sns.heatmap(correlations, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


#Python prefers to split data into Y and X
variables = num_data.drop(columns=['Default'])
variables = sm.add_constant(variables)

#indep_vars = variables[['Income', 'All Debt', 'Revolving Utilization',
#                        'Accts Opened 12 Months', 'Hard Inquiry Last 12']]
#Decided to pull anything great than correlation > abs(0.10)

indep_vars = variables[['Income', 'All Debt',
                        'Accts Opened 12 Months', 'Hard Inquiry Last 12']]


#Build Model
dev_PD = sm.Logit(data['Default'], indep_vars).fit()
print(dev_PD.summary())

#Predict Y values
y_pred_probs = dev_PD.predict(indep_vars)

#AUC
auc_score = roc_auc_score(data['Default'], y_pred_probs)
print(f"ROC AUC Score: {auc_score:.4f}")

#How comfortable do you feel? --> Me, not great. Maybe 1/10 and
    #an uneasy feeling that something is wrong.


#New Lesson - Lesson 8, Variables and WOE
#Let's look at multicollinearity as we dropped 'Revolving Utilization' for some
    #reason even though the correlation with PD was high.
vif_data = pd.DataFrame()
vif_data['var_names'] = variables.columns

vif_data['vif'] = [variance_inflation_factor(variables.values, i)
                   for i in range(len(variables.columns))]

#VIF rule of thumb:
    #  ~ 1: No correlation with other predictors
    #  1-5: Mild to moderate correlation
    # > 10: Strong correlation --> multicollinear

#Ignoring the constant, all the other variables look fine.

#Potential Validation Findings ... explain VIF and how the selection of the
    #group of variables matters! Don't just throw everything in the test!!!

#Let's plot Revolving Utilization to see if there are any obvious issues.
    #Add the plot details like title and axis labels as the business will ask!
plt.scatter(data['Revolving Utilization'], data['Default'])
plt.title('Revolving Utilization')
plt.xlabel('Revolving Utilization')
plt.ylabel('Defaults')
plt.show()

#The scatter doesn't tell us much, let's look at the distribution.
plt.hist(data['Revolving Utilization'])
plt.title('Revolving Utilization')
plt.xlabel('Revolving Utilization')
plt.ylabel('Count')
plt.show()

#The business has requested 'Revolving Utilization' so I am going to continue
    #to dig deeper.

#I am also uncomfortable with the first model we built. We will NOT look at
    #OOT until multiple models are built. It could allow human data leakage!!!

#WOE - weight of evidence creates numeric values with decimals for the Default
    #(dependent variable). This allows us to see the linearity/shape and can
    #handle missing values and outliers.
    #For more info see my video ---- https://youtu.be/6crUY-eHpjw?si=4R4Y0xuuKjZ0yMQO

#There are more complex and appropriate WOE methods however I want you to
    #understand the fine-tuning process. And yes, even with proprietary tools
    #you should manually check and adjust if necessary!!!
    #This can be very time consuming, which is why we first blind bin variables
        #and then manually bin ones that look more promising.
#df['group'] = pd.qcut(df['values'], q=4, labels=False)

data['Revolving Utilization_group'] = pd.qcut(data['Revolving Utilization'], 10, labels=None)


#Create a WOE dataframe for calcs.
woe_temp = pd.DataFrame()
total = len(data['Default'])
woe_temp['bads_p'] = data.groupby('Revolving Utilization_group').apply(
    lambda data: (data['Default'].sum() / total),
    include_groups=False)

woe_temp['goods_p'] = data.groupby('Revolving Utilization_group').apply(
    lambda data: ((data['Default'] == 0).sum() / total),
    include_groups=False)

#Save the actual number of goods and bads for later manual calculations.
woe_temp['bads_ct'] = data.groupby('Revolving Utilization_group').apply(
    lambda data: (data['Default'] == 1).sum(),
    include_groups=False)

woe_temp['goods_ct'] = data.groupby('Revolving Utilization_group').apply(
    lambda data: (data['Default'] == 0).sum(),
    include_groups=False)

woe_temp['var_woe'] = np.log(woe_temp['goods_p'] / woe_temp['bads_p'])

#Calculate IV
woe_temp['IV_woe'] = (woe_temp['goods_p'] - woe_temp['bads_p']) * woe_temp['var_woe']

#Since we have zeros in a few bins...the math of dividing by zero errors. We could manually fill
    #these with some very small number HOWEVER these binds should be combined!!!

#Since zeros are only in the bads_p column, we'll try and combine bins with goods_p that are
    #close to the next bin.
        #Bin 3 --> 0.07 is closest to 0.096
        #Bin 5 --> 0.126 is closest to 0.096
        #RESULT --> Combine bins 3, 4, and 5
    #Out of full transparency, I usually build out a complex WOE function at
        #each firm or they have their own tool. I am going to manually Excel
        #as I have done so when I am in a hurry and don't have a proper function.
   
#To reduce errors, I will save the final data inside a csv and then import it.
Revolving_Utilization_woe = pd.read_csv(r'C:\Users\Bianco\Documents\GitHub\Python_YouTube\Revolving Utilization WOE.csv')
