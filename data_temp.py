


#-------------------------------------
# Import libraries
#-------------------------------------

import csv
import xlrd
import numpy
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from numpy import loadtxt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler


#---------------------------------
# Data Analysis
#---------------------------------

filename = 'D:/Users/Mariano/Documents/Downloads/archive/BankChurners.csv'

data = read_csv(filename)

for col in data.columns: 
    print(col) 

peek = data.head(20)
print(peek)


shape = data.shape
print(shape)


types = data.dtypes
print(types)


description = data.describe()
print(description)


class_counts = data.groupby('Gender').size()
print(class_counts)


correlations = data.corr(method='pearson')
print(correlations)


skew = data.skew()
print(skew)


sh = data.shape[0]
print(sh)


#number of features
ft = data.shape[1]
print(ft)


#features list
feature_names = data.columns.tolist()

for column in feature_names:
    print(column)
    print(data[column].value_counts(dropna=False))
    
    
#-------------------------------------
# Working with features
#-------------------------------------

data['Card_Category'].value_counts(dropna=False)

# encode categorical column

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(data['Card_Category'])
label_encoder.transform(data['Card_Category'])


#-------------------------------
# Data plotting  
#-------------------------------

# Histogramas
data.hist()
pyplot.show()

# Densisty plots
data.plot(kind='density', subplots=True, sharex=False)
pyplot.show()

# box plots 
data.plot(kind='box', subplots=True, sharex=False, sharey=False)
pyplot.show()

correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
#ticks = numpy.arange(0,9,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
pyplot.show()


correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()


scatter_matrix(data)
pyplot.show()

#-------------------------------
# Data preparation  
#-------------------------------

# Rescale Data
array = data.values

# separate array into input and output components
X = array[:,0:len(data.columns)]
Y = array[:,len(data.columns)-1]

scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX = scaler.fit_transform(X)





#-------------------------------
# Data preparation  
#-------------------------------


# Correlation of data


# Determine possible target variable


# Prepare data for no supervised analysis


# Prepare data for PCA 


# Prepare categorical data as nummy


# Prepare data for entering a machine learning model 


# Prepare data for timeseries analysis 


