import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.mldata import fetch_mldata 
import sys
def lire_data(adr):
    f = open(adr,"r")
    e = f.readline()
    nbEx = eval(e)
    X = np.zeros([nbEx,2])
    Y = np.zeros(nbEx)
    for i in range(nbEx):
        x = f.readline()
        x = eval(x)
        X[i] = x[0]
        Y[i] = x[1]
    f.close()
    return X, Y


"""Notre MLP n'arrive pas à obtenir mieux que le score obtenu avec un perceptron à noyau Gaussien car il s'agit d'un ensemble de ensemble de perceptron non Gaussien 
qui ne peuvent pas individuellement atteindre le niveau du perceptron Gaussien'"""
def gen1(nbEx, noise=0):
    X = np.random.rand(nbEx,2)-0.5
    y = [max(X[i]) <= 0.35for i in range(nbEx)]
    
    for i in range (nbEx):
        if random.random() < noise:
            y[i] = 1- y[i]
    return X,y
    
def exo1():
  noise = 0.0
  X,y = lire_data("learn.data")
  X_train,X_test,y_train,y_test= train_test_split( X,y,test_size=0.3,random_state=random.seed() )
  clf = MLPClassifier(hidden_layer_sizes=(50,50,50), alpha = 0.0001, solver = "lbfgs",activation = "tanh")

  clf.fit(X_train,y_train)

  print(clf.coefs_, clf.intercepts_)
  print(clf.score(X_test,y_test))

def ex2():
  noise = 0.0
  X,y = gen1(100, noise)
  X_test,y_test = gen1(1000, noise)
  clf = MLPClassifier(hidden_layer_sizes=(3), alpha = 0.001, solver = "lbfgs",activation = "logistic")
  clf.fit(X,y)
  print(clf.score(X_test,y_test))
  
  X = X*1000
  X_test = X_test*1000
  # les coordonn´ees de chaque exemple sont multipli´ees par 1000
  # cela ne devrait rien changer pour l’apprentissage
  clf.fit(X,y)
  print(clf.score(X_test,y_test))
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)
  X_test = scaler.transform(X_test)
  clf.fit(X,y)
  print(clf.score(X_test,y_test))


def exo3():
  """"L execution de ce programme nous donne une hyberbole en sortie """
  X, Z = lire_data("TP3.data1")
  def f1(x):
    return x**2
  def f2(x):
    return math.sin(x)
  def f3(x):
    return abs(x)
  def f4(x):
    return max(min(math.ceil(x),1),0)

  l_f = [f1, f2, f3, f4]
  
  for f in l_f:
    y = [f(x) for x in Z]
    clf = MLPRegressor(hidden_layer_sizes=(3,),solver="lbfgs",activation="tanh",learning_rate ="adaptive")
    clf.fit(X, y)
    yy = clf.predict(X)
    plt.scatter(Z,y)
    plt.plot(Z,yy, label = "Cible")
    plt.legend()
    plt.show()


#Avec les paramètres suivants, on obtient un score environant de 0.935
#hidden_layer_sizes=(50,50,50), alpha = 0.0001, solver = "lbfgs",activation = "tanh"

#Avec uniquement une couche de 5, le score obtenu est de 0.676

#Avec 3 couches de 5, score de 0.56

#En utilisation l'activation en logistic, et 3 couches de 50, le score obtenu est de 0.90
#Avec une seule couche, score de 0.92 (Logistic prefere utiliser moins de couches?))
mns = fetch_mldata("MNIST original")
noise = 0.0
X= mns.data
Y= mns.target
activate = str(sys.argv[2])#input()#logistic
hidden = sys.argv[1]#input(#(25)

X_train,X_test,y_train,y_test= train_test_split( X,Y,test_size=0.3,random_state=random.seed() )
clf = MLPClassifier(hidden_layer_sizes=(hidden), alpha = 0.0001, solver = "sgd",activation = activate)

clf.fit(X_train,y_train)

print(clf.coefs_, clf.intercepts_)
print(clf.score(X_test,y_test))










