import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
from plotfunctions import demographical_data, numerical_data, pay_hist_plot, scatter_plot_numerical_data
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Trying to set the seed
seed = 209
np.random.seed(seed)
import random
random.seed(seed)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
#print(df.describe().to_latex())

num_features = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
cat_features = ['SEX', 'EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

demographical_features = ['AGE','SEX','EDUCATION','MARRIAGE']
demographical_data(df,demographical_features)
bill_amt_features = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_amt_features = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
#numerical_data(df,bill_amt_features)





pay_features = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
#pay_hist_plot(df,pay_features)

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



bill_amt_cutoffs = np.zeros((2,6))
col = ''
print('Column   lower_cutoff    higher_cutoff')
for i in range(6):
    col = 'BILL_AMT' + str(i+1)
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
for i in range(6):
    col = 'PAY_AMT' + str(i+1)
   # df[col] = pd.to_numeric(df[col])
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    pay_amt_cutoffs[0,i] = Q1 - 1.5*IQR
    pay_amt_cutoffs[1,i] = Q3 + 1.5*IQR
    print(col+'     '+str(pay_amt_cutoffs[0,i])+'    '+str(pay_amt_cutoffs[1,i]))

#scatter_plot_numerical_data(df,bill_amt_features,cutoffs = bill_amt_cutoffs,add_cutoff=200000)
#scatter_plot_numerical_data(df,pay_amt_features,cutoffs = pay_amt_cutoffs,add_cutoff = 200000)

#Removing outliers from BILL_AMTX and PAY_AMTX
#outlier_limit used as adjustment for rule of thumb..
outlier_limit = 225000
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

df = df.drop(df[(df.PAY_AMT1 < (pay_amt_cutoffs[0,0])) |
                (df.PAY_AMT2 < (pay_amt_cutoffs[0,1])) |
                (df.PAY_AMT3 < (pay_amt_cutoffs[0,2])) |
                (df.PAY_AMT4 < (pay_amt_cutoffs[0,3])) |
                (df.PAY_AMT5 < (pay_amt_cutoffs[0,4])) |
                (df.PAY_AMT6 < (pay_amt_cutoffs[0,5]))].index)
#Dropping instances where pay is -2 as this case is not defined in the dataset
df = df.drop(df[(df.PAY_0 == -2) |
                (df.PAY_2 == -2) |
                (df.PAY_3 == -2) |
                (df.PAY_4 == -2) |
                (df.PAY_5 == -2) |
                (df.PAY_6 == -2)].index)
'''
print(df.describe())
n, bins, patches = plt.hist(df.defaultPaymentNextMonth, bins='auto', color='#0504aa', rwidth=1)
plt.grid(axis='y')
#plt.xlabel('Age')
plt.xticks(df['defaultPaymentNextMonth'].unique())
plt.title('Default')
plt.show()
'''


# Always scale before onehotencoding if using columntransfer
sc = StandardScaler()


#df[num_features] = pd.DataFrame(scaler.fit_transform(df))
#data_df = df.copy()

data_df = pd.get_dummies(df, columns = cat_features)
features = data_df[num_features]
#scaler = sc.fit(features.values)
#features = scaler.transform(features.values)
#data_df[num_features] = features

print(data_df[num_features].describe())
# Features and targets
X = data_df.loc[:, data_df.columns != 'defaultPaymentNextMonth'].values
y = data_df.loc[:, data_df.columns == 'defaultPaymentNextMonth'].values

#Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,shuffle=True)
scaler = sc.fit(X_train[:,:14])
X_train[:,:14] = scaler.transform(X_train[:,:14])
X_test[:,:14] = scaler.transform(X_test[:,:14])


#Resampling
n_defaults = np.count_nonzero(y_train==1)
def_samples = 0
nondef_samples = 0
i = 0
j = 0
X_train_new = np.zeros((n_defaults*2,X_train.shape[1]))
y_train_new = np.zeros((n_defaults*2,1))
print(X_train_new.shape)
print(y_train_new.shape)
print('ndflts',n_defaults)
while def_samples < n_defaults:
    if(y_train[i] == 1 and def_samples < n_defaults):
        X_train_new[j,:] = X_train[i,:]
        y_train_new[j] = y_train[i]
        def_samples = def_samples + 1
        j = j+1
    elif(y_train[i] == 0 and nondef_samples < n_defaults):
        X_train_new[j,:] = X_train[i,:]
        y_train_new[j] = y_train[i]
        nondef_samples = nondef_samples + 1
        j = j+1
    i = i + 1

#X_train = X_train_new
#y_train = y_train_new



sizes = [30, 50, 50, 1]
etas = np.logspace(-5, 1, 10)
lamb = np.logspace(-5, 1, 10)
# lamb = np.zeros(1)

def bestCurve(y):
	defaults = sum(y == 1)
	total = len(y)
	x = np.linspace(0, 1, total)
	y1 = np.linspace(0, 1, defaults)
	y2 = np.ones(total-defaults)
	y3 = np.concatenate([y1,y2])
	return x, y3




#grid_search(logit,X_train,y_train,X_test,y_test,sizes,etas,lamb)

sizes = [X_train.shape[1], 70, 50, 1]
#sizes = [X_train_new.shape[1], 70, 50, 1]

etas = np.logspace(-5, 1, 7)
lambdas = [np.logspace(-5, 2, 8)]
etas = np.append(etas,[11,12])
print(lambdas)

#lamb = np.logspace(-5, 2, 8)

#accuracies = np.zeros((len(etas),len(lambdas)))
#accuracies = grid_search(NeuralNetwork,X_train,y_train,X_test,y_test,sizes=sizes,etas=etas,lamdbdas = lambdas)
#accuracies = grid_search(NeuralNetwork,X_train_new,y_train_new,X_test,y_test,sizes,etas,lambdas)

#accuracies = np.zeros((len(etas),1))
#accuracies = grid_search(logit,X_train,y_train,X_test,y_test,sizes=sizes,etas=etas,lamdbdas=lambdas)
#accuracies = grid_search(logit,X_train_new,y_train_new,X_test,y_test,sizes,etas,lamdbdas = lambdas)
accuracies = test_keras(X_train,y_train,X_test,y_test,sizes,etas,lambdas)


print(np.amax(accuracies))
print("\nMax element : ", np.argmax(accuracies))
# returning Indices of the max element
# as per the indices
print("\nIndices of Max element : ", np.argmax(accuracies, axis=0))
print("\nIndices of Max element : ", np.argmax(accuracies, axis=1))
lambdas = []
# lambda = 10 eta = 0.01 sgd
# λ=0.0001andη=0.1 NR
# λ=0.1andη=0.0001 Vanilla
# $\lambda$ = 10 and $\eta$ = 0.001. Neural Network

'''


#Optimal Conditions
#SGD $\eta$ = 0.1 and $\lambda$ = 0.001.
#VANILLA $\eta$ = 0.01 $\lambda$ = 0.1
#NEURAL NET eta = 0.001, lamb  = 10

#Resampled
#SGD $\eta$ = 1  $\lambda$ = 0.01
#VANILLA $\eta$ = 0.1 $\lambda$ = 0.01
#NEURAL NET eta = 0.001, lamb  = 10

logreg_sgd = logit(X_train, y_train, epochs=100, batch_s=100, eta_in=0.001, type='SGD')
logreg_van = logit(X_train, y_train, epochs=100, batch_s=100, eta_in=0.0001, type='Vanilla')
neural = NeuralNetwork(X_train, y_train, sizes=sizes, epochs=20, batch_size=500, eta=0.001,lmbd=10)

#probabilities = Object.predict(X_test, classify=True)
#y_pred = logreg.predict(X_test,classify=True)
#print(f1_score(y_test, y_pred))


y_probas_sgd = logreg_sgd.predict(X_test)
y_probas_sgd2 = np.zeros(y_probas_sgd.shape)
for i in range(len(y_probas_sgd)):
    y_probas_sgd2[i] = 1 - y_probas_sgd[i,0]
y_probas_sgd = np.c_[y_probas_sgd2,y_probas_sgd]


y_probas_van = logreg_van.predict(X_test)
y_probas_van2 = np.zeros(y_probas_van.shape)
for i in range(len(y_probas_van)):
    y_probas_van2[i] = 1 - y_probas_van[i,0]
y_probas_van = np.c_[y_probas_van2,y_probas_van]


y_probas_neu, loss  = neural.predict(X_test,y_test)
y_probas_neu2 = np.zeros(y_probas_neu.shape)
for i in range(len(y_probas_neu)):
    y_probas_neu2[i] = 1 - y_probas_neu[i,0]
y_probas_neu = np.c_[y_probas_neu2,y_probas_neu]

x_bc = np.zeros((n_defaults,1))
y_bc = np.zeros(x_bc.shape)
fig, axes = plt.subplots(1, 2)
axes[0, 0] = skplt.metrics.plot_cumulative_gain(y_test, y_probas_sgd)
axes[0, 1].scatter(x, y)


fig = skplt.metrics.plot_cumulative_gain(y_test, y_probas_sgd)
skplt.metrics.plot_cumulative_gain(y_test, y_probas_van)

x_bc,y_bc = bestCurve(y_test)
plt.plot(x_bc,y_bc)
plt.show()
#probabilities = logreg.predict(X_test, classify=True)
#print(probabilities.shape)

#fig, ax = plt.subplots()
#ax.hist(probabilities-y_test, bins=[0, 0.5, 0.5, 1])
#ax.set_xticks(np.arange(0, 1, step=0.5))
# ax.set_xticklabels(0.5,1)

#plt.show()
'''