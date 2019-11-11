import pandas as pd
import os
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score



# Trying to set the seed
np.random.seed(0)
import random
random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)



df.rename(index=str, columns={"default payment next month": "targets"}, inplace=True)


#Assume that pay 0 is actaul meant to be pay




print(df.columns)

#There is no pay_1 -->


# Features and targets
X = df.loc[:, df.columns != 'targets'].values
y = df.loc[:, df.columns == 'targets'].values


# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough").fit_transform(X)


print(X)



