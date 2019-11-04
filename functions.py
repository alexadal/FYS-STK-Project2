import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import random
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from keras.utils import to_categorical
from tensorflow.python.framework import ops
ops.reset_default_graph()
random.seed(209)
np.random.seed(100)

"""
--------------------------------------------------------------------
Class test function
--------------------------------------------------------------------
"""

def grid_search(class_input,X_train,y_train,X_test,y_test,sizes,etas,lamdbdas,MSE=False):
    accuracies = np.zeros((len(etas),len(lamdbdas)))
    #loss = np.zeros((len(etas), len(lamdbdas)))
    sns.set()
    title = ''
    cmap = ''
    prob =[]
    for i,eta_in in enumerate(etas):
        for j,lamb in enumerate(lamdbdas):
            print("Eta",eta_in)
            #NN = NeuralNetwork(X_train,y_train,sizes,epochs=100, batch_size=50,eta=eta_in)
           #Object = class_input(X_train,y_train,epochs=100, batch_s=100,eta_in=eta_in,lamd=lamb)
            if class_input == NeuralNetwork:
                if MSE:
                    print("MSE = true")
                    Object = class_input(X_train, y_train, sizes, epochs=100, batch_size=100, eta=eta_in, cost=MSE_Cost)
                    #probabilities, loss = Object.predict(X_test, y_test, classify=False)
                    probabilities, loss = Object.predict(X_test, FrankeFunc(X_test[:,1],X_test[:,2]), classify=False)
                    print("Loss", loss)
                    accuracies[i][j] = loss
                    prob.append(probabilities)
                    title = 'MSE Neural Network - Franke function'
                    cmap = 'viridis_r'
                else:
                    Object = class_input(X_train, y_train, sizes, epochs=50, batch_size=50, eta=eta_in)
                    probabilities, loss = Object.predict(X_test, y_test, classify=True)
                    print("Loss", loss)
                    accuracies[i][j] = accuracy_score_numpy(probabilities, y_test)
                    title = 'Accuracy Neural Network'
                    cmap = 'viridis'


            #Object.activations()
            elif class_input == logit:
                Object = class_input(X_train, y_train, epochs=100, batch_s=100, eta_in=eta_in, lamd=lamb,type='SGD')
                probabilities = Object.predict(X_test, classify=True)
                print("Probabilities", probabilities)
                accuracies[i][j] = accuracy_score_numpy(probabilities, y_test)
                title = 'Accuracy Logistic Regression'
                cmap = 'viridis'

    print("Accuracies",accuracies)
    z_max = np.amax(y_train)
    prob_max = np.amax(prob)
    min_run = np.amin(accuracies)
    print("Probabilities_max",prob_max)
    print("Y_train", z_max)
    print("Minimum MSE", min_run)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(accuracies, annot=True, ax=ax, cmap=cmap, vmax = 0.15)
    ax.set_title("Accuracy")
    #ax.set_ylim(etas[-1],etas[0])
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    #ax.set_yticks(etas)
    ax.set_yticklabels(etas)
    plt.ylim((etas[0],len(etas)))
    plt.title(title)
    plt.show()



def validation_test(class_input,X_train,y_train,X_valid,y_valid,sizes,etas,lamdbdas):
    sizes = [30,1,1]
    loss = []

    #Max 4 hidden layers with 10 Neurons test
    for i in range (1,5):
        for j in range (1,51):
            for h in range(1,len(sizes)-1):
                print(h)
                sizes[h] = j
                print("Sizes",sizes)

            Object = class_input(X_train, y_train, sizes, epochs=100, batch_size=50, eta=etas[6],lmbd=lamdbdas[4])
            prob,l = Object.predict(X_valid, y_valid, classify=True)
            if ~np.isnan(l):
                loss.append([l,i,j])

        sizes.insert(-1,1)
    print("Loss Matrix ", loss)
    min_loss = np.amin(loss)
    index = find(loss,min_loss)
    print("Index",index)
    print("Minimum loss is {} for {} hidden layers and {} Neurons".format(loss[index[0][0]][0],loss[index[0][0]][1], loss[index[0][0]][2]))
    return loss







def check_hist(class_input,X_train,y_train,X_test,y_test,sizes,eta,lamdbda):

        Object = class_input(X_train, y_train, sizes, epochs=100, batch_size=100, eta=etas[5],lmbd=lamb[4])
        probabilities,loss = Object.predict(X_test, y_test, classify=True)
        print(probabilities.shape)
        fig, ax = plt.subplots()
        counts, bins, patches = ax.hist(probabilities, 2,facecolor='deepskyblue', edgecolor='gray')
        ax.set_xticks(bins.round(0))
        ax.set_xticklabels(bins, rotation=0, rotation_mode="anchor", ha="right")
        plt.show()
        return 0





















"""
--------------------------------------------------------------------
Functions used to generate data for testing
--------------------------------------------------------------------
"""

#Generate data for testing
def generate_data():
    #np.random.seed(0)
    #X, y = datasets.make_moons(200, noise=0.20)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


def load_regdata():
    terrain = imread("SRTM_data_Norway_25.tif")
    n = len(terrain[0])
    m = len(terrain[:,1])
    x = np.linspace(0,50*m,m)
    y = np.linspace(0,50*n,n)
    X,Y = np.meshgrid(x,y)

    z = terrain.ravel()
    x = X.ravel()
    y = Y.ravel()

    x_deg = np.c_[x, y]
    poly = PolynomialFeatures(degree=1)
    X_ = poly.fit_transform(x_deg)
    print("Regdata",X_)
    return X_,z

def gen_frake(N,sigma):
    N = N
    x = np.arange(0, 1, 1 / N)
    y = np.arange(0, 1, 1 / N)
    X, Y = np.meshgrid(x, y)
    X = X.ravel()
    Y = Y.ravel()
    z = FrankeFunc(X, Y)
    error = np.random.normal(0, sigma, size=z.shape)
    z = z + error
    z.ravel()
    x_deg = np.c_[X, Y]
    poly = PolynomialFeatures(degree=1)
    X_ = poly.fit_transform(x_deg)
    print("Franke", X_.shape)
    return X_, z


def FrankeFunc(x, y):

    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4



""" 
--------------------------------------------------------------------
General functions
--------------------------------------------------------------------
"""

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    exp_term = np.exp(x)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)

def linear(x):
    return x

def linear_der(x):
    return np.ones(x.shape)

def ReLU(x):
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass
            else:
                x[i][k] = 0
    return x


def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def find(searchList, elem):
    index = []
    indElem = elem
    resultList = []
    for ind in range(0, len(searchList)):
        if searchList[ind][0] == elem:
            resultList.append(ind)
    index.extend([resultList])
    return index

def accuracy_score_numpy(Y_test, Y_pred):
    Y_pred = Y_pred.reshape(Y_test.shape)
    #print('Shape comparision  {}  {}'.format(Y_test.shape,Y_pred.shape))
    return np.sum(Y_test == Y_pred) / len(Y_test)


"""
--------------------------------------------------------------------
Classes used
--------------------------------------------------------------------

"""

class logit:


    def __init__(self,X,y,epochs=10,batch_s=80,eta_in=0.1,lamd=0,type='Vanilla'):
        self.weights = self.rnd_betas(len(X[0,:]))
        self.gd_fit(X,y,type,epochs,batch_s,eta=eta_in)


    def linreg(self,X,betas):
        return np.dot(X,betas)

    #give
    def pi_beta(self,beta_x):
        return 1/(1+np.exp(-beta_x))


    # Create cost function based on the number of parameters
    def cost_func(self,X,y,betas):
        #get number of observation to average cost function over
        n = len(X[0,:])
        scores = self.linreg(X,betas)
        Cost = np.sum(y*scores-np.log(1+np.exp(scores)))
        return Cost

    #Initiate random betas for grad descent
    def rnd_betas(self, m):
        beta = np.random.randn(m, 1)
        return beta

    #Get the gradient of log function
    def gd(self,X,y,betas):
        n = len(X[0])
        #grad = 1/n*np.sum((y-self.pi_beta(X, betas)*X))
        #print("P-matrix",self.pi_beta(X, betas).shape)
        scores = self.linreg(X,betas)
        prob = self.pi_beta(scores)
        #print("prob shape",prob.shape)
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
    def SGD(self, X_train, y_train, epochs, m_batch_size, eta=0.1):
        n=0

        n=len(X_train[:])
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
                self.update_m_batch(batchx,batchy,eta,epoch,m_batch_size,k)
                k+=1



    def learning_schedule(self,t):
        t0, t1 = 5, 50
        return t0 / (t + t1)


    def update_m_batch(self,batchx,batchy,eta,epoch,m,i):
        gradients = self.gd(batchx,batchy,self.weights)
        if eta ==0:eta= self.learning_schedule((epoch*m+i))
        #print("SGD eta",eta)
        self.weights -= eta*gradients


    #Normal Gradient Desent --> choose learning rate as constant or Newtons method
    def gd_fit(self,X,y, iter='Vanilla',epochs=10,batch_s = 50,conv=10**-8,eta =0.1):
        gamma = []
        betas_old = []
        Niterations = 1000
        n = len(X[:])
        #betas = self.rnd_betas(n)
        for i in range(Niterations):
            betas_old = self.weights
            if iter == 'NR':
                eta_NR = self.NR(X, betas_old)
                if i == 0:
                    eta_old = self.NR(X, betas_old)*0
                else:
                    eta_old = eta_NR
                #betas-= eta@self.gd(X,y,betas)
                self.weights -= eta_NR@self.gd(X,y,self.weights)
                #Break loop for convergence
                if i>0:
                    if np.abs(np.mean(np.mean(eta_NR,axis=1),axis=0)-np.mean(np.mean(eta_old,axis=1),axis=0)) <= conv:
                        break
            elif iter == 'Vanilla':
                self.weights -= eta*self.gd(X,y,betas_old)
            elif iter == 'SGD':
                self.SGD(X,y,epochs,batch_s,eta)
                break

        return self.weights

    def predict(self, X,classify=False):
        a = np.dot(X,self.weights)
        prob = sigmoid(a)

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
Create different cost-function classes
That when called within another object, gives inheritance to the attributes
1. Logit versions also known as Cross-Entropy Cost
2. MSE version also known as Quadratic Cost

Use classes and @staticmethod to avoid many different functions and easy the change of cost funtion 
within the Neural Net class

Note that the scaling is simplified as this does not change the algebraic behaviour to find minima

"""
#Remember to update activation in for additional class here
class CrossE_Cost:

    @staticmethod
    def cost_f(a, y):
        """Use np.nan_to_num to avoid nan of log part if a,y = 1 """
        return (-1./len(y))*np.sum(y*np.log(a)+(1-y)*np.log(1-a))
    @staticmethod
        #Y included for easy running of Neural Net Code
        #delta = dC/da_out, sigmoid part cancels increasing learning pace
    def delta(z, a, y):
        return(a-y)


class MSE_Cost:

    @staticmethod
    def cost_f(a, y):
        #print("MSE cost used")
        return (0.5/len(y))*np.sum((a-y)**2)
    @staticmethod
        #z included for easy running of Neural Net Code
        #delta = dC/da_out, sigmoid
    def delta(z, a, y):
        #print("MSE loss used")
        return(a-y)


#Xavier & He
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
        self.activations = self.activations_in(cost, sizes)
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
        self.activations = self.activations_in(cost,sizes)
        #print("last activation",self.activations[-1])

    def activations_in(self,cost,sizes):
        activations_l = []
        if cost == CrossE_Cost:
            for i in range(len(sizes)-1):
                activations_l.append(sigmoid)
                #print("Ok")
            if sizes[-1] == 1:
                activations_l.append(sigmoid)
            else:
                #softmax
                activations_l.append(softmax)
        elif cost == MSE_Cost:
            for i in range(len(sizes)-1):
                activations_l.append(linear)
            activations_l[-1] = ReLU
            print(activations_l)


        return activations_l

    def create_biases_and_weights(self,zeros=True):
        #self.weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        #self.biases = np.zeros(self.n_hidden_neurons) + 0.01

        if zeros:self.biases = [np.zeros((y)) for y in self.sizes[1:]]
        else:self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        #Note x & y changed to use np.dot instead of matmul
        #Connections between prev layer with X number of neurons and next layer of neurons

        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        #print(self.weights)


    #Simple feed-forward algorithm
    #def feed_forward(self,a):
        #Return activiation with a as input
     #   for bias, weight in zip(self.biases, self.weights):
      #      a = sigmoid(np.dot(weight, a)+bias)
       # return a


    def backpropagate(self,X,y):
        """Return derivative [der_b,der_w] of cost function w.r.t weigts and bias
        --> out for first of arrays these
        """
        #Initialize array
        der_b = [np.zeros(bias.shape) for bias in self.biases]
        der_w = [np.zeros(weight.shape) for weight in self.weights]
        #Initialize activation arrays
        #X = X.reshape(-1,1)
        activation = X
        #print("Activation",activation.shape)
        #Store all activations and zs regardless of layer
        #activations = [x]
        activations = [X]
        zs = []

        #feed-forward
        for act,bias, weight in zip(self.activations,self.biases, self.weights):
            #print("Wt",weight.shape)

            z = np.matmul(activation,weight.T)+bias
            #print("Z",z.shape)
            zs.append(z)
            activation = act(z)
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


        delta = self.cost.delta(zs[-1],activations[-1],y.reshape(-1,1))

        der_w[-1] = np.matmul(delta.T,activations[-2])
        der_b[-1] = np.sum(delta,axis=0)

        #Start with next to out-layer
        for l in range(2,self.n_layers):
            z = zs[-l]
            #print("Z",z.shape)
            #sig_der = sigmoid_der(z)
            act_der = sigmoid_der(z)

            if self.activations[-l] == ReLU:
                act_der = reluDerivative(z)
                print("-------------------------------------------------------------- Der ReLU -------------------------------------------------------")


            elif self.activations[-l] == linear:
                act_der = linear_der(z)
                print("-------------------------------------------------------------- Der Linear -------------------------------------------------------")

            #For layer h
            #delta_h = dC/d_ah X da_h/dz_h
            # 1. dC/da_h = delta_(h+1)*w_h+1
            # 2. da_h/dz_h = sigma'(z_h)
            delta = np.matmul(delta,self.weights[-l+1])*act_der
            #print("Activations",activations[-l-1].T)
            #print("Weights",self.weights[-l+1].T.shape)
            #print("Activations",activations[0])
            der_w[-l] = np.matmul(delta.T,activations[-l-1])
            der_b[-l] = np.sum(delta,axis=0)
            #print("Derivate",der_w)

        return der_b, der_w

        # Se nærmere på hvordan X er ift vector
    def SGD(self, X, y, epochs, m_batch_size, eta=0.1):
        n = len(X[:])

        for epoch in range(epochs):
            # Pass på axis
            print("Epoch no: " + "{}".format(epoch))
            randomize = np.arange(len(X[:,0]))

            np.random.shuffle(randomize)

            x_rand = X[randomize]
            y_rand = y[randomize]
            #print("batch size",m_batch_size)
            mini_batches_x = [x_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            mini_batches_y = [y_rand[m:m + m_batch_size] for m in range(0, n, m_batch_size)]
            k = 0
            for batchx, batchy in zip(mini_batches_x, mini_batches_y):
                self.update_m_batch(batchx, batchy, epoch, eta)
                k = k+1
                print("Batch no: " + "{}".format(k))
                cost = self.cost.cost_f(self.feed_forward_out(batchx),batchy.reshape(-1,1))
                print("Loss",cost)



    def learning_schedule(self, t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    #Update with regularization l2,l1
    def update_m_batch(self, batchx, batchy, epoch, eta_i, m=100, i=1,lamd=0):
        #gradients = self.gd(batchx, batchy, self.weights)

        #der_b = [np.zeros(bias.shape) for bias in self.biases]
        #der_w = [np.zeros(weight.shape) for weight in self.weights]
        eta = 0

        if eta_i == 0:
            eta = self.learning_schedule((epoch * m + i))
        eta = eta_i


        #Get dCx/dw & dCx/db

        der_b,der_w = self.backpropagate(batchx, batchy)
        self.biases = [b-(eta)*db for b,db in zip(self.biases,der_b)]
        #self.biases = self.biases - (eta / m) * der_b
        self.weights = [(1-(eta*lamd))*w-(eta)*dw for w,dw in zip(self.weights,der_w)]




    def feed_forward_out(self,X):
        #Return activiation with a as input
        a = X
        z_h = []
        probabilities = np.zeros(len(X[:]))
        loss = 0
        for bias, weight, activation in zip(self.biases, self.weights,self.activations):

            z = np.matmul(a,weight.T)+bias

            #Enumerate & softmax
            a = activation(z)
            if activation == linear:
                print("-------------------------------------------------------------- Linear -------------------------------------------------------")

            probabilities = a



            #probabilities[i] = a

            #print(probabilities[i])

        return probabilities


    def predict(self, X,y,classify=False):
        #print("W_out",self.biases)
        prob = self.feed_forward_out(X)
        classification = prob
        print("Probabilities",prob)
        loss = self.cost.cost_f(prob, y.reshape(-1, 1))
        if classify:
            for i in range(len(prob)):
                if prob[i] >= 0.5:
                    classification[i] = 1
                    #print(True)
                else:
                    classification[i] = 0
                    #print(False)


        print("Loss",loss)
        return classification, loss

class k_NN():
    def __init__(
            self,
            X,
            y,
            sizes,
            cost=CrossE_Cost,
            epochs=100,
            batch_size=100,
            eta=0,
            lmbd=0.000):
        self.X = X
        self.y = y
        self.sizes = sizes
        #self.cost = cost
        #self.activations = self.activations_in(cost, sizes)
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
        self.Object = self.create_neural_network_keras(eta,lmbd)
        self.Object.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        #scores = DNN.evaluate(X_test, Y_test)


    def create_neural_network_keras(self, eta,lmbd):
        model = Sequential()
        model.add(Dense(self.sizes[1],input_dim=self.sizes[0],activation='sigmoid', kernel_regularizer=l2(lmbd)))
        for neurons in self.sizes[2:-1]:
            print("Neurons",neurons)
            model.add(Dense(neurons, activation='sigmoid', kernel_regularizer=l2(lmbd)))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=eta)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        #print(model.summary())
        return model

    def predict(self, X,y,classify=False):
        scores = self.Object.evaluate(X, y)
        return scores












"""
--------------------------------------------------------------------
Init tester
--------------------------------------------------------------------
"""


#Create unit test for functions file
if __name__ == "__main__":

    X, y = generate_data()
    #X,y = load_regdata()
    #X,y = gen_frake(50,1)

    print("Shape X",X.shape)
    #y = to_categorical(y)
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.2,shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_t, y_t, test_size = 0.2,shuffle=True)

    print("Franke",FrankeFunc(X_test[:,1],X_test[:,2]).shape)
    print(X_test.shape)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)




    sizes = [30,20,20,1]
    print(len(sizes))
    etas = np.logspace(-5,  1, 7)
    #etas = np.logspace(1,  1, 10)
    lamb = np.logspace(-5, 1, 7)
    #etas = etas[:4]
    print("etas",etas)
    #lamb = np.zeros(1)
    #Run test funtion
    #grid_search(logit,X_train,y_train,X_test,y_test,sizes,etas,lamb)
    #grid_search(NeuralNetwork,X_train,y_train,X_train,y_train,sizes,etas,lamb)
    #grid_search(NeuralNetwork,X_train,y_train,X_test,y_test,sizes,etas,lamb,MSE=True)
    k_Object = k_NN(X_train,y_train,sizes)
    DNN_k = np.zeros((len(etas), len(lamb)), dtype=object)

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lamb):
            DNN = k_NN(X_train,y_train,sizes,eta=eta,lmbd=lmbd)
            scores = DNN.predict(X_test, y_test)

            DNN_k[i][j] = DNN

            print("Learning rate = ", eta)
            print("Lambda = ", lmbd)
            print("Test accuracy: %.3f" % scores[1])
            print()

    sns.set()

    train_accuracy = np.zeros((len(etas), len(lamb)))
    test_accuracy = np.zeros((len(etas), len(lamb)))

    for i in range(len(etas)):
        for j in range(len(lamb)):
            DNN = DNN_k[i][j]

            train_accuracy[i][j] = DNN.predict(X_train, y_train)[1]
            test_accuracy[i][j] = DNN.predict(X_test, y_test)[1]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()



    #check_hist(NeuralNetwork, X_train, y_train, X_test, y_test, sizes, etas[5], lamb[3])

    #losses = validation_test(NeuralNetwork, X_train, y_train, X_valid, y_valid, sizes, etas, lamb)
    #losses = validation_test(NeuralNetwork, X_train, y_train, X_train, y_train, sizes, etas, lamb)
    #losses = validation_test(NeuralNetwork, X_train, y_train, X_test, y_test, sizes, etas, lamb)

    #min_loss = np.amin(losses)
    #print(min_loss)

    """                             
    
    Valid - Minimum loss is 4.5780359721437545e-05 for 4 hidden layers and 40 Neurons

    
    Train - Minimum loss is 0.00014233529772095604 for 4 hidden layers and 30 Neurons
    
    
    """



