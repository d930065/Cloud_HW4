from pyspark.mllib.classification import LogisticRegressionWithSGD, LabeledPoint
import numpy as np
import math
from pyspark import SparkConf, SparkContext

sc = SparkContext("local", "First App")

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ] # need floats
    
    return LabeledPoint(label, features)

def train(X_train, Y_train):
    epochs = 300
    b = 0.0
    w = np.zeros(4)
    
    lr = 0.04
    b_lr = 0
    w_lr = np.ones(4)

    for epoch in range(epochs):
        #z = np.dot(X_train, w) + b
        z = X_train.map(lambda x: np.dot(x, w) + b)
        
        #pred = sigmoid(z)
        pred = z.map(lambda x: 1 / (1.0 + math.exp(-x)))
        #loss = y_train - pred
        loss = Y_train.zip(pred).map(lambda y: y[0] - y[1])
        
        #b_grad = -1 * np.sum(loss)
        b_grad = loss.map(lambda l: -1 * l).reduce(lambda x,y : x+y)
        
        
        #w_grad = -1 * np.dot(loss, X_train)
        w_grad = X_train.zip(loss).map(lambda x: -1 * x[0] * x[1]).reduce(lambda x,y: x+y)

        #b_lr += b_grad**2
        b_lr += b_grad**2
        #w_lr += w_grad**2
        w_lr += w_grad**2

        b = b - lr/np.sqrt(b_lr) * b_grad
        
        #w = w - lr/np.sqrt(w_lr)*w_grad
        w = w - lr/np.sqrt(w_lr) * w_grad
        if (epoch+1) % 10 == 0:
            print('Current epoch: {}\nTraining Error: {}\n'.format(epoch+1,test(X_train, Y_train, w, b)))            
    return w, b

def test(X_train, Y_train, w, b):
    #z = np.dot(X_train, w) + b
    z = X_train.map(lambda x: np.dot(x, w) + b)        
    #pred = sigmoid(z)    
    pred = z.map(lambda x: 0 if (1 / (1.0 + math.exp(-x)) < 0.5) else 1)
    labelsAndPreds = pred.zip(Y_train).map(lambda x: (x[0],x[1]))
    trainErr = labelsAndPreds.filter(lambda seq: seq[0] != seq[1]).count() / float(parsedData.count())
    return(trainErr)
    
# Load and parse the data
data = sc.textFile("data_banknote_authentication.txt")
parsedData = data.map(mapper)
X_train = parsedData.map(lambda x: np.array(x.features))
Y_train = parsedData.map(lambda x: x.label)
w, b = train(X_train, Y_train)
