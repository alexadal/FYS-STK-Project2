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
from sklearn.utils.multiclass import unique_labels


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


#The following function is borrowed from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
      #     yticks=np.arange(cm.shape[0]),
           yticks = [0,1],
           ylim = [-0.5,1.5],
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     #        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax