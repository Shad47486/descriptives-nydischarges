# importing modules from part 1 and 2 and other needed ones
from lib2to3.pgen2.pgen import DFAState
import math
import statistics
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn 
import researchpy as rp
import matplotlib.pyplot as plt
from tableone import TableOne, load_dataset
from statsmodels.formula.api import ols
# Question im looking to answer from this descriptive statistics is to see what gender on average has a higher los? 

#load in dataset (since its too big using api due to always crashing)
df = pd.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')
df
#checking data
df.shape
df.columns

# Descriptive Stats

meanlos = np.mean(df['length_of_stay'])
meanlos
medianlos = np.median(df['length_of_stay'])
medianlos

# We can see that the los is 3.446 from line 24 
# and a median of 3 
# But we are looking for the los for genders respectively
# So we have to create new dataframes with just those columns 

fcol = df[df['gender'] == 'F']
fcol
mcol = df[df['gender'] == 'M']
mcol 

female_col = df[df['gender'] == 'F']['length_of_stay']
male_col = df[df['gender'] == 'M']['length_of_stay']
male_col
stats.ttest_ind(female_col, male_col)

gender_mean = pd.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')
model = ols('length_of_stay ~ gender', gender_mean).fit()
print (model.summary())

female_mean = pd.DataFrame({'gender': df['Gender'], 'LOS': ''})

df2 = df.groupby('gender')
for gender, value in df2['length_of_stay']: 
    print((df2, value.mean())) 
#codes 51-53 did not work was having trouble printing a table for it

#attempt at using tableone 
columns = ['gender', 'length_of_stay']
categorical = ['gender']
groupby = ['gender']
nonnormal = ['length_of_stay']
mytable = TableOne(df, columns=columns, categorical=categorical, groupby=groupby, nonnormal=nonnormal, pval=False)
print(mytable.tabulate(tablefmt = "fancy_grid"))
