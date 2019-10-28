import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import sys
import matplotlib.pyplot as plt
random.seed(209)
np.random.seed(100)

from sklearn.linear_model import LogisticRegression


"""
--------------------------------------------------------------------
Class test function
--------------------------------------------------------------------
"""

def grid_search(class_input,sizes,X_test,y_test,etas,lamdbdas):
    accuracies = np.zeros((len(etas),len(lamdbdas)))
    sns.set()

    for i,eta_in in enumerate(etas):
        for j,lamb in enumerate(lamdbdas):
            print("Eta",eta_in)
            #NN = NeuralNetwork(X_train,y_train,sizes,epochs=100, batch_size=50,eta=eta_in)
            Object = class_input(X_train,y_train,sizes,epochs=100, batch_size=50,eta=eta_in,lmbd=lamb)
            probabilities = Object.predict(X_test,classify=True)
            #print("Probabilities", probabilities)
            #print("Fasit",y)
            accuracies[i][j] = accuracy_score_numpy(probabilities,y_test)
            #print("Accuracy",accuracies[i])


    print("Accuracies",accuracies)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(accuracies, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


"""
--------------------------------------------------------------------
Functions used to generate data for testing
--------------------------------------------------------------------
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
--------------------------------------------------------------------
General functions
--------------------------------------------------------------------
"""

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


"""
--------------------------------------------------------------------
Classes used
--------------------------------------------------------------------

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
                self.SGD(X,y,100,int(len(X[:])/100),eta_fixed=True)
                break

        return self.weights


    #Need to create predict
    #def fit(self,X_train,y_train):



"""
Create different cost-function classes
That when called within another object, gives inheritance to the attributes
1. Logit versions also known as Cross-Entropy Cost
2. MSE version also known as Quadratic Cost

Use classes and @staticmethod to avoid many different functions and easy the change of cost funtion 
within the Neural Net class

Note that the scaling is simplified as this does not change the algebraic behaviour to find minima

"""
class CrossE_Cost:
    @staticmethod
    def cost_f(a, y):
        """Use np.nan_to_num to avoid nan of log part if a,y = 1 """
        return -np.sum(np.nan(y*np.log(a)+(1-y)*np.log(1-a)))
    @staticmethod
        #Y included for easy running of Neural Net Code
        #delta = dC/da_out, sigmoid part cancels increasing learning pace
    def delta(z, a, y):
        return(a-y)


class MSE_Cost:
    @staticmethod
    def cost_f(a, y):
        return 0.5*(a-y)**2
    @staticmethod
        #Y included for easy running of Neural Net Code
        #delta = dC/da_out, sigmoid
    def delta(z, a, y):
        return(a-y)*sigmoid(z)



class NeuralNetwork:
    #Sizes is an array containing layers of corresponding neurons
    #Tune lambda for Regularization
    def __init__(
            self,
            X,
            y,
            sizes,
            cost=CrossE_Cost,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.000):
        self.X = X
        self.y = y
        self.sizes = sizes
        self.cost = cost
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
        self.create_biases_and_weights()
        self.SGD(X,y,epochs,batch_size,eta)

    def create_biases_and_weights(self,zeros=False):
        #self.weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        #self.biases = np.zeros(self.n_hidden_neurons) + 0.01

        if zeros:self.biases = [np.zeros(y, 1) for y in self.sizes[1:]]
        else:self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        #Note x & y changed to use np.dot instead of matmul
        #Connections between prev layer with X number of neurons and next layer of neurons

        self.weights = [0.001*np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        #print(self.weights)


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
        der_b = [np.zeros(bias.shape) for bias in self.biases]
        der_w = [np.zeros(weight.shape) for weight in self.weights]
        #Initialize activation arrays
        X = X.reshape(-1,1)
        activation = X
        #Store all activations and zs regardless of layer
        #activations = [x]
        activations = [X]
        zs = []

        #Activate all layers
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias
            zs.append(z)
            activation = sigmoid(z)
            #print("Activations Back",activation)
            #np.append(activations,activation)
            activations.append(activation)

        """Now we pass backward using the chain rule to find the respective derivatives
        
        C = Cost function, cost function is an object that can be changed and has its own function
        delta_out = dC/da_out*da_out/dz
        
        - Start with computing derivatives of dC/db_out and dC/dw_out = delta_out*dz/dw_out
        For backprop do not make distintion on dela_out as it is passed back into the layers in for 
        loop
        Reach end of array with [-1]
        
        """

        delta = self.cost.delta(zs[-1],activations[-1],y)

        der_w[-1] = np.dot(delta,activations[-2].T)
        der_b[-1] = delta

        #Start with next to out-layer
        for l in range(2,self.n_layers):
            z = zs[-l]
            #print("Z",z.shape)
            sig_der = sigmoid_der(z)
            #print("Z-der", sigmoid_der(z).shape)
            #For layer h
            #delta_h = dC/d_ah X da_h/dz_h
            # 1. dC/da_h = delta_(h+1)*w_h+1
            # 2. da_h/dz_h = sigma'(z_h)
            delta = np.dot(self.weights[-l+1].T,delta)*sig_der
            #print("Activations",activations[-l-1].T)
            #print("Weights",self.weights[-l+1].T.shape)
            #print("Activations",activations[0])
            der_w[-l] = np.dot(delta,activations[-l-1].T)
            der_b[-l] = delta
            #print("Derivate",der_w)

        return der_b, der_w

        # Se nærmere på hvordan X er ift vector
    def SGD(self, X, y, epochs, m_batch_size, eta=0.1):
        n = len(X[:])

        for epoch in range(epochs):
            # Pass på axis
            randomize = np.arange(len(X[:,0]))

            np.random.shuffle(randomize)

            x_rand = X[randomize]
            y_rand = y[randomize]
            #print("batch size",m_batch_size)
            mini_batches_x = [x_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            mini_batches_y = [y_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            k = 0

            for batchx, batchy in zip(mini_batches_x, mini_batches_y):
                #print("Batch no", k+1)

                self.update_m_batch(batchx, batchy, epoch, eta)
                k += 1


    def learning_schedule(self, t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    #Update with regularization l2,l1
    def update_m_batch(self, batchx, batchy, epoch, eta_i, m=100, i=1,lamd=0):
        #gradients = self.gd(batchx, batchy, self.weights)

        der_b = [np.zeros(bias.shape) for bias in self.biases]
        der_w = [np.zeros(weight.shape) for weight in self.weights]


        if eta_i == 0:
            eta = self.learning_schedule((epoch * m + i))
        eta = eta_i
        #print("SGD eta", eta)
        #print("M",m)
        #self.weights -= eta * gradients

        #print("eta_out",eta)

        for x,y in zip(batchx,batchy):
            #Get dCx/dw & dCx/db

            delta_der_b, delta_der_w = self.backpropagate(x, y)
            # Calculate SUM_x(dCx/dw) & SUM_x(dCx/db)
            der_b = [db + d_db for db,d_db in zip(der_b,delta_der_b)]
            der_w = [dw + d_dw for dw, d_dw in zip(der_w, delta_der_w)]

        self.biases = [b-(eta/m)*db for b,db in zip(self.biases,der_b)]
        self.weights = [(1-(eta*lamd/m))*w-(eta/m)*dw for w,dw in zip(self.weights,der_w)]




    def feed_forward_out(self,X):
        #Return activiation with a as input
        a = 0
        z_h = []
        probabilities = np.zeros(len(X[:]))
        for i in range(len(X[:])):
            X_in = X[i,:]
            X_in.reshape(-1,1)
            """
            for bias, weight in zip(self.biases, self.weights):
                #print("Sample",activation)
                z_h = sigmoid(np.dot(weight, activation)+bias)
                activation = z_h
            a_h = z_h.T
            """
            a_h = self.feed_forward(X_in).T
            a = np.dot(self.weights[-1],a_h)+self.biases[-1]
            probabilities[i] = sigmoid(a)
            #probabilities[i] = a

            #print(probabilities[i])

        return probabilities


    def predict(self, X,classify=False):
        prob = self.feed_forward_out(X)
        classification = prob
        if classify:
            for i in range(len(prob)):
                if prob[i] >= 0.5:
                    classification[i] = 1
                    #print(True)
                else:
                    classification[i] = 0
                    #print(False)
        return classification



"""
--------------------------------------------------------------------
Init tester
--------------------------------------------------------------------
"""


#Create unit test for functions file
if __name__ == "__main__":

    num_observations = 1000
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    X = np.vstack((x1, x2)).astype(np.float32)
    y= np.hstack((np.zeros(num_observations),np.ones(num_observations)))
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


    logreg = logit(X,y)
    #print("Beta_vec",betas.shape)
    #print("Betas", logreg.gd_fit(X,y,iter='NR'))
    #clf = LogisticRegression(fit_intercept=True, C=1e15)
    #clf.fit(X[:,1:], y.ravel())
    #print("SKlearn results",clf.intercept_, clf.coef_)



    sizes = [3,50,1]
    etas = np.logspace(-5, -1, 20)
    lamb = np.logspace(-5, -1, 7)
    #Run test funtion
    grid_search(NeuralNetwork,sizes,X_test,y_test,etas,lamb)

    sns.set()
    test_accuracy = np.zeros((len(etas), len(lamb)))

"""
sizes = [2,3,1]


weights = []
#X = number of neurons prev layer
#Y = number of neurons new layer
for x, y in zip(sizes[:-1], sizes[1:]):
    print("XY",y,x)
    weights = np.random.randn(y,x)
    print(weights)

"""

