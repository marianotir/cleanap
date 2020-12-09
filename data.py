

#-------------------------------------
# Import libraries
#-------------------------------------

import csv
import xlrd
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from numpy import loadtxt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer


#---------------------------------
# Data Analysis
#---------------------------------


# Load data
filename = 'D:/Users/Mariano/Documents/Downloads/archive/BankChurners.csv'
data_orig = read_csv(filename)

df = data_orig.copy()

# Check consistency of duplicates
duplicates = len(df[df.duplicated() == True])
if(duplicates>0):
    print(" Duplicates founds and deleted ")
    df = df[df.duplicated() == False]
    
# Eliminate client identificator column => needs to be improved for generalization
#df.drop(['CLIENTNUM'], axis=1, inplace=True)
df["CLIENTNUM"] = df.index + 1
df.rename(columns={'CLIENTNUM': 'Key'}, inplace=True)

# Change column names to make them more visual
cols = list(df.columns.values)   
index = 1                                             #start at 1
for c in range(len(cols)-1):
    cols[index] = "X_"+str(index)                    #rename the column name based on index
    index += 1  

cols[len(cols)-1] = "Y"                                           #add one to index

df.columns = cols

# drop columns with the same value
df = df.drop(df.std()[(df.std() == 0)].index, axis=1)

# drop column similar to target => needs to be improved for generalization
df.drop(['X_21'], axis=1,inplace=True)

# head
peek = df.head(20)
print(peek)

# shape
shape = df.shape
print(shape)

# types
types = df.dtypes
print(types)

# description
description = df.describe()
print(description)

# split categorical , numerical , target key
df_key_target = df[['Key', 'Y']]

df_temp = df.drop(df.columns[[0, 21]], axis=1)

df_num = df_temp.select_dtypes(include=np.number)

df_cat = df_temp.select_dtypes(include=['object'])


# clean dataframe using imputer 

#numerical imputer ::: simple way df_mean_imputed = df.fillna(df.mean())
imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_fit = imp_num.fit_transform(df_num)

df_num = pd.DataFrame(imp_fit,columns=df_num.columns)

# categorical imputer
imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp_fit = imp_cat.fit_transform(df_cat)

df_cat = pd.DataFrame(imp_fit,columns=df_cat.columns)


# clean from outliers
def Remove_Outlier_Indices(df):
    Q1 = df.quantile(0.04)
    Q3 = df.quantile(0.96)
    IQR = Q3 - Q1
    trueList = ~((df < (Q1 - 2.5 * IQR)) |(df > (Q3 + 2.5 * IQR)))
    return trueList

# Index List of Non-Outliers
nonOutlierList = Remove_Outlier_Indices(df_num)

# Non-Outlier Subset of the Given Dataset
df_num = df_num[nonOutlierList]

print('number of outliers detected:', df_num.isna().values.sum() )

# Concat datasets
df_clean = pd.concat([df_key_target, df_num,df_cat], axis=1)

# drop outliers
df_clean.dropna(inplace = True) 


# create app


# continue with featingapp


# Plan
# Generate el html 














