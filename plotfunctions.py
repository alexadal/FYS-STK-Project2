import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import logit, grid_search
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def demographical_data(df = None,columns = None):
    fig, axs = plt.subplots(2, 2)
    k = 0
    for i in range(2):
        for j in range(2):
            col = columns[k]
            axs[i, j].hist(df[col], bins='auto', color='#000000', rwidth=4)
            axs[i, j].set_title(col)
            if col != 'AGE':
                axs[i, j].set_xticks(df[col].unique())
            k = k + 1
    fig.tight_layout()
    plt.show()

def numerical_data(df = None, columns = None, cutoffs = None):
    fig, axs = plt.subplots(2, 3)
    k = 0
    for i in range(2):
        for j in range(3):
            col = columns[k]
            axs[i, j].hist(df[col], bins='auto', color='#0504aa', rwidth=1)
            if(cutoffs != None):
                axs[i, j].vlines(cutoffs[0, k], 0, 7000)
                axs[i, j].vlines(cutoffs[1, k], 0, 7000)
            axs[i, j].set_title(col)
            k = k + 1
    fig.tight_layout()
    plt.show()


def pay_hist_plot(df = None, columns = None):
    fig, axs = plt.subplots(3, 2)
    k = 0
    for i in range(3):
        for j in range(2):
            col = columns[k]
            axs[i, j].hist(df[col], bins='auto', color='#0504aa', rwidth=1)
            axs[i, j].set_xticks(df[col].unique())

            axs[i, j].set_title(col)
            k = k + 1
    fig.tight_layout()
    plt.show()

def scatter_plot_numerical_data(df = None, columns = None, cutoffs = None,add_cutoff = 0):
    ind = [int(i) for i in df.index.values]
    fig, axs = plt.subplots(2, 3)
    k = 0
    for i in range(2):
        for j in range(3):
            col = columns[k]
            axs[i, j].scatter(ind, df[col], facecolors='none', edgecolors='r',s = 0.5)
            #if cutoffs != None:
            axs[i, j].hlines(cutoffs[0, k], 0, max(ind))
            axs[i, j].hlines(cutoffs[1, k], 0, max(ind))
            axs[i, j].hlines(cutoffs[1, k] + add_cutoff, 0, max(ind),color='b')
            axs[i, j].set_title(col)
            k = k + 1
    fig.tight_layout()
    plt.show()