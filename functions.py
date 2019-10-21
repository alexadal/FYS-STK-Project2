import pandas as pd
import numpy as np
from sklearn import datasets
import random
import matplotlib.pyplot as plt
random.seed(209)

from sklearn.linear_model import LogisticRegression



#Generate data for testing
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


class logit:

    def linreg(self,X,betas):
        return np.dot(X,betas)

    #give
    def pi_beta(self,beta_x):
        return 1/(1+np.exp(-beta_x))



    # Create cost function based on the number of parameters
    def cost_func(self,X,y,betas):
        #get number of observation to average cost function over
        n = len(X[0])
        scores = self.linreg(X,betas)
        Cost = np.sum(y*scores-np.log(1+np.exp(scores)))
        return Cost

    #Initiate random betas for grad descent
    def rnd_betas(self, m):
        beta = 0.001*np.random.randn(m, 1)
        return beta

    #Get the gradient of log function
    def gd(self,X,y,betas):
        n = len(X[0])
        #grad = 1/n*np.sum((y-self.pi_beta(X, betas)*X))
        #print("P-matrix",self.pi_beta(X, betas).shape)
        scores = self.linreg(X,betas)
        prob = self.pi_beta(scores)
        s_p = y.reshape(prob.shape)-prob

        grad = -np.dot(X.T,s_p)
        #grad = -X.T @y+X.T@self.pi_beta(X, betas)
        return grad

    def NR(self,X,betas):
        #W = np.zeros((len(X[:, 0]), len(X[:, 0])))
        scores = self.linreg(X,betas)
        prob = self.pi_beta(scores)
        p_0 = np.ones((len(X[:, 1]), 1)) - prob
        p_1 = prob
        w = p_1 @ p_0.T
        W = np.diag(np.diag(w))
        A = X.T @ W @ X
        u, s, v = np.linalg.svd(A)
        A_inv = np.dot(v.transpose(), np.dot(np.diag(s ** -1), u.transpose()))
        eta = A_inv
        return eta

    #def st_gd(self,X,y,betas):
        






    #Normal Gradient Desent --> choose learning rate as constant or Newtons method
    def gd_fit(self,X,y, iter='Vanilla',conv=0.000001):
        Niterations = 0
        gamma = []
        betas_old = []
        Niterations = 30000
        eta = 0
        n = len(X[0])
        betas = self.rnd_betas(n)
        for i in range(Niterations):
            betas_old = betas
            print("eta",eta)
            if iter == 'NR':
                eta_old = eta
                eta = self.NR(X, betas_old)
                betas-= eta@self.gd(X,y,betas)
                if i>0:
                    if np.abs(np.mean(np.mean(eta,axis=1),axis=0)-np.mean(np.mean(eta_old,axis=1),axis=0)) <= conv:
                        break
            elif iter == 'Vanilla':
                eta = 5*10**(-5)
                betas -= eta*self.gd(X,y,betas_old)

        return betas



    #def fit(self,X_train,y_train):





#Create unit test for functions file
if __name__ == "__main__":

    print("test")
    """
    X, y = generate_data()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    """
    """
    data = load_data("marks.csv", None)
    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]
    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]
    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]
    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

    # preparing the data for building the model

    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    """
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    y= np.hstack((np.zeros(num_observations),np.ones(num_observations)))

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    print("X",X)
    y = y.ravel()
    n = len((X[0]))
    #betas = np.zeros((X.shape[1],1))
    logreg = logit()
    betas = logreg.rnd_betas(n)
    logreg.cost_func(X , y,betas)
    grad = logreg.gd(X,y,betas)
    print("Beta_vec", betas.shape)
    #print("Gamma", logreg.gd_fit(X,y,iter='NR').shape)
    print("Gradient", grad.shape)
    #gamma = logreg.gd_fit(X,y,iter='NR')
    #step = gamma@grad
    print("Beta_vec",betas.shape)
    print("Betas", logreg.gd_fit(X,y,iter='NR'))
    print("Gradient",grad.shape)

    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(X[:,1:], y.ravel())

    print("SKlearn results",clf.intercept_, clf.coef_)


    #print(step.shape)
"""
[ 1043.15877062]
 [-1043.98909762]]
Gradient (3, 1)

Betas [[6664.01682755]
 [3234.95056589]
 [1727.37221704]]


"""