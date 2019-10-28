import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Trying to set the seed
np.random.seed(209)
import random
random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)


# Remove instances with zeros only for past bill statements or paid amounts
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

'''
#Renaming to male and female
#df.SEX.replace(to_replace=1, value='Male',inplace=True)
#df.SEX.replace(to_replace=2, value='Female',inplace = True)

n, bins, patches = plt.hist(df.SEX, bins='auto', color='#0504aa', rwidth=1)
plt.grid(axis='y')
plt.xlabel('Value')
# plt.ylabel('Frequency')
plt.title('SEX')
# plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

#calculate cutoff for plots
Q3 = df.AGE.quantile(0.75)
Q1 = df.AGE.quantile(0.25)
IQR = Q3-Q1

cutoff = Q3+1.5*IQR

n, bins, patches = plt.hist(df.AGE, bins='auto', color='#0504aa', rwidth=1)
plt.grid(axis='y')
plt.xlabel('Age')
plt.vlines(cutoff,0,1450)
# plt.ylabel('Frequency')
plt.title('AGE')
# plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

# atempting scatterplot
#a = plt.scatter(df.index,df.AGE)
#plt.show()
#df.plot.scatter(df.ID,df.AGE)
ind = [None] * 28497
for i in range(28497):
    ind[i] = i
plt.scatter(ind,df.AGE)
plt.show()
#df.plot.scatter(df.index,df.AGE)
'''
#calculate and print cutoff for pay_amt
bill_amt_cutoffs = np.zeros((2,6))
col = ''
print(bill_amt_cutoffs.shape)

print('Column   lower_cutoff    higher_cutoff')
for i in range(1,6):
    col = 'BILL_AMT' + str(i)
   # df[col] = pd.to_numeric(df[col])
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    bill_amt_cutoffs[0,i] = Q1 - 1.5*IQR
    bill_amt_cutoffs[1,i] = Q3 + 1.5*IQR
    print(col+'     '+str(bill_amt_cutoffs[0,i])+'    '+str(bill_amt_cutoffs[1,i]))

pay_amt_cutoffs = np.zeros((2,6))
col = ''
#calculate and print cutoff for bill amt
for i in range(1,6):
    col = 'PAY_AMT' + str(i)
   # df[col] = pd.to_numeric(df[col])
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    pay_amt_cutoffs[0,i] = 0
    pay_amt_cutoffs[1,i] = Q3 + 1.5*IQR
    print(col+'     '+str(pay_amt_cutoffs[0,i])+'    '+str(pay_amt_cutoffs[1,i]))

'''
df['grad_school']=(df['EDUCATION']==1).astype('int')
df['university']=(df['EDUCATION']==2).astype('int')
df['high_school']=(df['EDUCATION']==3).astype('int')
df['others_education']=(df['EDUCATION']==4).astype('int')
df['unknown_education']=(~df['EDUCATION'].isin([1,2,3,4])).astype('int')


df['male']=(df['SEX']==1).astype(int)
df['female']=(df['SEX']==0).astype(int)

df['married']=(df['MARRIAGE']==1).astype(int)
df['single']=(df['MARRIAGE']==2).astype(int)
df['other_marriage']=(~df['MARRIAGE'].isin([1,2])).astype(int)
df.drop(['SEX','EDUCATION','MARRIAGE'],axis=1,inplace=True)

'''

'''
fig, axs = plt.subplots(2, 3)
k = 0
for i in range(2):
    for j in range(3):
        col = 'BILL_AMT' + str(k+1)
        axs[i, j].hist(df[col], bins='auto', color='#0504aa', rwidth=1)
        axs[i, j].vlines(bill_amt_cutoffs[0,k],0,7000)
        axs[i, j].vlines(bill_amt_cutoffs[1,k],0,7000)
        axs[i, j].set_title(col)
        k = k+1

plt.show()


fig, axs = plt.subplots(2, 3)
k = 0
for i in range(2):
    for j in range(3):
        col = 'PAY_AMT' + str(k+1)
        axs[i, j].hist(df[col], bins='auto', color='#0504aa', rwidth=1)
        axs[i, j].vlines(pay_amt_cutoffs[0,k],0,500)
        axs[i, j].vlines(pay_amt_cutoffs[1,k],0,500)
        axs[i, j].set_title(col)
        k = k+1

plt.show()

fig, axs = plt.subplots(2, 3)
k = 0
for i in range(2):
    for j in range(3):
        col = 'BILL_AMT' + str(k+1)
        axs[i, j].scatter(ind,df[col],facecolors='none', edgecolors='r')
        axs[i, j].hlines(bill_amt_cutoffs[0,k],0,max(ind))
        axs[i, j].hlines(bill_amt_cutoffs[1,k],0,max(ind))
        axs[i, j].set_title(col)
        k = k+1

plt.show()

'''

#Removing outliers from BILL_AMTX and PAY_AMTX
#outlier_limit used as adjustment for rule of thumb..
outlier_limit = 220000
df = df.drop(df[(df.BILL_AMT1 > (bill_amt_cutoffs[1,0]+outlier_limit)) |
                (df.BILL_AMT2 > (bill_amt_cutoffs[1,1]+outlier_limit)) |
                (df.BILL_AMT3 > (bill_amt_cutoffs[1,2]+outlier_limit)) |
                (df.BILL_AMT4 > (bill_amt_cutoffs[1,3]+outlier_limit)) |
                (df.BILL_AMT5 > (bill_amt_cutoffs[1,4]+outlier_limit)) |
                (df.BILL_AMT6 > (bill_amt_cutoffs[1,5]+outlier_limit))].index)

df = df.drop(df[(df.BILL_AMT1 < (bill_amt_cutoffs[0,0])) |
                (df.BILL_AMT2 < (bill_amt_cutoffs[0,1])) |
                (df.BILL_AMT3 < (bill_amt_cutoffs[0,2])) |
                (df.BILL_AMT4 < (bill_amt_cutoffs[0,3])) |
                (df.BILL_AMT5 < (bill_amt_cutoffs[0,4])) |
                (df.BILL_AMT6 < (bill_amt_cutoffs[0,5]))].index)
                
outlier_limit = 200000
df = df.drop(df[(df.PAY_AMT1 > (pay_amt_cutoffs[1,0]+outlier_limit)) |
                (df.PAY_AMT2 > (pay_amt_cutoffs[1,1]+outlier_limit)) |
                (df.PAY_AMT3 > (pay_amt_cutoffs[1,2]+outlier_limit)) |
                (df.PAY_AMT4 > (pay_amt_cutoffs[1,3]+outlier_limit)) |
                (df.PAY_AMT5 > (pay_amt_cutoffs[1,4]+outlier_limit)) |
                (df.PAY_AMT6 > (pay_amt_cutoffs[1,5]+outlier_limit))].index)
'''
for i in range(1,6):
    col = 'BILL_AMT' + str(i)
   # df[col] = pd.to_numeric(df[col])
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    bill_amt_cutoffs[0,i] = Q1 - 1.5*IQR
    bill_amt_cutoffs[1,i] = Q3 + 1.5*IQR
    print(col+'     '+str(bill_amt_cutoffs[0,i])+'    '+str(bill_amt_cutoffs[1,i]))

print(str((max(df.PAY_AMT1))))
fig, axs = plt.subplots(2, 3)
k = 0
for i in range(2):
    for j in range(3):
        col = 'PAY_AMT' + str(k+1)
        axs[i, j].hist(df[col], bins='auto', color='#0504aa', rwidth=1)
        #axs[i, j].vlines(pay_amt_cutoffs[0,k],0,500)
        #axs[i, j].vlines(pay_amt_cutoffs[1,k],0,500)
        axs[i, j].set_title(col)
        k = k+1

plt.show()
'''
cat_features = ['SEX', 'EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
print(df[cat_features[:4]].describe())
print(df[cat_features[4:]].describe())

print(min(df.defaultPaymentNextMonth))
data_df = pd.get_dummies(df, columns = cat_features)
print(data_df.describe())
# Features and targets
X = data_df.loc[:, data_df.columns != 'defaultPaymentNextMonth'].values
y = data_df.loc[:, data_df.columns == 'defaultPaymentNextMonth'].values
# Categorical variables to one-hot's
#onehotencoder = OneHotEncoder(categories="auto")
#X = ColumnTransformer([("", onehotencoder, [3]),],remainder="passthrough").fit_transform(X)
#X = ColumnTransformer([("", onehotencoder, []),],remainder="passthrough").fit_transform(X)

#print(X[1,:])
#Testing

logreg = logit(X,y)
#betas = logreg.rnd_betas(n)
    #logreg.cost_func(X , y,betas)
    #grad = logreg.gd(X,y,betas)
    #print("Beta_vec", betas.shape)
    #print("Gamma", logreg.gd_fit(X,y,iter='NR').shape)
    #print("Gradient", grad.shape)
    #gamma = logreg.gd_fit(X,y,iter='NR')
    #step = gamma@grad
    #print("Beta_vec",betas.shape)
'''
#atempting to run logistic regression . . .
print("Betas", logreg.gd_fit(X,y.ravel(),iter='SGD'))


    #print("Gradient",grad.shape)
X = data_df.loc[:, data_df.columns != 'defaultPaymentNextMonth'].values
y = data_df.loc[:, data_df.columns == 'defaultPaymentNextMonth'].values
clf = LogisticRegression(fit_intercept=True, C=1e15, max_iter=1000,random_state=209)
clf.fit(X, y.ravel())
print("SKlearn results",clf.intercept_, clf.coef_)

'''
