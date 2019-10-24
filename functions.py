import pandas as pd
import numpy as np
from sklearn import datasets
import random
import sys
import matplotlib.pyplot as plt
random.seed(209)

from sklearn.linear_model import LogisticRegression

"""
Functions used to generate data for testing
"""


#Generate data for testing
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


"""
Classes used

"""

class logit:


    def __init__(self,X,Y):
        self.weights = self.rnd_betas(len(X[0]))


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

    #Se nærmere på hvordan X er ift vector
    def SGD(self,X_train,y_train,epochs,m_batch_size,eta_fixed=True,X_test=None,y_test=None):
        n=0
        if X_test: n = len(X_test[:])
        else:n=len(X_train[:])
        for epoch in range(epochs):
            #Pass på axis
            randomize = np.arange(len(X_train[:]))
            np.random.shuffle(randomize)
            x_rand = X_train[randomize]
            y_rand = y_train[randomize]

            mini_batches_x = [x_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            mini_batches_y = [y_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            k = 0
            for batchx,batchy in zip(mini_batches_x,mini_batches_y):
                #print("Batch no", k+1)
                self.update_m_batch(batchx,batchy,eta_fixed,epoch,m_batch_size,k)
                k+=1



    def learning_schedule(self,t):
        t0, t1 = 5, 50
        return t0 / (t + t1)


    def update_m_batch(self,batchx,batchy,eta_fixed,epoch,m,i):
        gradients = self.gd(batchx,batchy,self.weights)
        eta = 0
        if eta_fixed: eta=0.001
        else:eta= self.learning_schedule((epoch*m+i))
        #print("SGD eta",eta)
        self.weights -= eta*gradients


    #Normal Gradient Desent --> choose learning rate as constant or Newtons method
    def gd_fit(self,X,y, iter='Vanilla',conv=0.000001):
        Niterations = 0
        gamma = []
        betas_old = []
        Niterations = 1000
        eta = 0
        n = len(X[:])
        #betas = self.rnd_betas(n)
        for i in range(Niterations):
            betas_old = self.weights
            print("eta",eta)
            if iter == 'NR':
                eta_old = eta
                eta = self.NR(X, betas_old)
                #betas-= eta@self.gd(X,y,betas)
                self.weights -= eta@self.gd(X,y,self.weights)
                #Break loop for convergence
                if i>0:
                    if np.abs(np.mean(np.mean(eta,axis=1),axis=0)-np.mean(np.mean(eta_old,axis=1),axis=0)) <= conv:
                        break
            elif iter == 'Vanilla':
                eta = 5*10**(-5)
                self.weights -= eta*self.gd(X,y,betas_old)
            elif iter == 'SGD':
                print("Batch_size",n)
                self.SGD(X,y,2000,int(len(X[:])/10),eta_fixed=True)
                break

        return self.weights


    #Need to create predict
    #def fit(self,X_train,y_train):


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1-x)


"""
Create different cost-function classes
That when called within another object, gives inheritance to the attributes
1. Logit versions also known as Cross-Entropy Cost
2. MSE version also known as Quadratic Cost

Use classes and @staticmethod to avoid many different functions and easy the change of cost funtion 
within the Neural Net class

"""
class CrossECost:
    @staticmethod
    def cost_f(a, y):
        """Use np.nan_to_num to avoid nan of log part if a,y = 1 """
        return -np.sum(np.nan(y*np.log(a)+(1-y)*np.log(1-a)))
    @staticmethod
        #Y included for easy running of Neural Net Code
        #delta = dC/da_out, sigmoid part cancels increasing learning
    def delta(z, a, y):
        return(a-y)






class NeuraNetwork:
    #Sizes is an array containing layers of corresponding neurons
    #Tune lambda for Regularization
    def __init__(
            self,
            X_data,
            Y_data,
            sizes,
            epochs=10,
            batch_size=100,
            #eta=0.1,
            lmbd=0.0):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.sizes = sizes
        #self.n_inputs = X_data.shape[0]
        #self.n_features = X_data.shape[1]
        #self.n_hidden_neurons = n_hidden_neurons
        #self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        #self.iterations = self.n_inputs // self.batch_size
        #self.eta = eta
        self.lmbd = lmbd
        self.n_layers = len(sizes)
        self.weights = self.create_biases_and_weights()

    def create_biases_and_weights(self,zeros=False):
        #self.weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        #self.biases = np.zeros(self.n_hidden_neurons) + 0.01
        if zeros:self.biases = [np.zeros(y, 1) for y in range(self.sizes[1:])]
        else:self.biases = [np.random.randn(y, 1) for y in range(self.sizes[1:])]
        #Note x & y changed to use np.dot instead of matmul
        #Connections between prev layer with X number of neurons and next layer of neurons
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:1], self.sizes[:-1])]


    #Simple feed-forward algorithm
    def feed_forward(self,a):
        #Return activiation with a as input
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a)+bias)
        return a


    def backpropagate(self,X,y):
        """Return derivative [der_b,der_w] of cost function w.r.t weigts and bias
        --> out for first of arrays these
        """
        #Initialize array
        der_w = [np.zeros(bias.shape) for bias in self.biases]
        der_b = [np.zeros(weight.shape) for weight in self.weights]
        #Initialize activation arrays
        activation = x
        #Store all activations and zs regardless of layer
        activations = [x]
        zs = []

        #Activate all layers
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight,activation)+bias
            np.append(zs,z)
            activation = sigmoid(z)
            np.append(activations,activation)

        """Now we pass backward using the chain rule to find the respective derivatives
        
        C = Cost function, cost function is an object that can be changed and has its own function
        delta_out = dC/da_out*da_out/dz
        
        - Start with computing derivatives of dC/db_out and dC/dw_out = delta_out*dz/dw_out
        
        """




        for h in range(2,self.n_layers):






















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
    """
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    y= np.hstack((np.zeros(num_observations),np.ones(num_observations)))

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    print("X",len(X[:]))
    y = y.ravel()
    n = len((X[:,1]))
    #betas = np.zeros((X.shape[1],1))
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
    print("Betas", logreg.gd_fit(X,y,iter='SGD'))
    #print("Gradient",grad.shape)

    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(X[:,1:], y.ravel())

    print("SKlearn results",clf.intercept_, clf.coef_)


    #print(step.shape)
    """
"""
[ 1043.15877062]
 [-1043.98909762]]
Gradient (3, 1)

Betas [[6664.01682755]
 [3234.95056589]
 [1727.37221704]]


"""

sizes = [2,3,1]


weights = []
#X = number of neurons prev layer
#Y = number of neurons new layer
for x, y in zip(sizes[:-1], sizes[1:]):
    print("XY",y,x)
    weights = np.random.randn(y,x)
    print(weights)


