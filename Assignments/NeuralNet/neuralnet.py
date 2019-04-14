'''
Abdullah Zameek

8th April 2019 - Introduction to Neural Networks


Helper Math and Other Support functions.
The following helper functions will help with the linear algebra and other mathematical operations associated with the Neural Network, 
as well as with some file manipulation

'''
import math
import time
import numpy as np 
######################### MATH AND HELPER FUNCTIONS START HERE ######################################

#Given two vectors of equal length, this functions computes the sum of the two and returns it
def vectorAdd(x,y):
    sum = []
    if(len(x) != len(y)):
        print("The lists have to be of equal length!")
        return None
    else:
        sum = [i+j for i,j in zip(x,y)]
        return sum

#Given two vectors of equal length, this function computes the dot/inner product of the two
def dotProduct(x,y):
    if(len(x) != len(y)):
        print("The lists have to be of equal length!")
        return None
    else:
        return sum(i*j for i,j in zip(x,y))

#Given a vector, this function mulitplies said vector with a scalar quantity 
def scalarMultiply(scalar,vect):
    return [scalar*i for i in vect]

#This function creates a vector of n-dimensions and fills it with zeroes
def zeroVec(numZeroes):
    zeroVec = []
    zeroVec = [0]*numZeroes
    return zeroVec

#This function converts a file to a list of lists. Used to access the csv's with ease
def ftol(filename):
    stream = open(filename, "r")
    returnList = []
    for elem in stream:
        elem = elem.split()
        for item in elem:
            item = item.split(',')
        floats = list(map(float, item))
        returnList.append(floats)
    return returnList

#This function used to pre-process the y values
def labelPreprocessor(y):
    newLabelList = []
    for elem in y: 
        for num in elem:
            num = int(num) - 1
            newLabelList.append(num)
    return newLabelList


###############################   END OF MATH AND OTHER FUNCTIONS ###########################################


class NeuralNetwork():

    def __init__(self):
        pass
    
    #sigmoid function
    def g(self, z):
        return 1/(1 + math.e**((z*-1)))

    # A single neural unit, with an option of whether or not to include a sigmoid function 
    def neuralUnit(self,wi, a, sigmoid=True):
        b = wi[0]
        z = dotProduct(a,wi[1:]) + b
        if(sigmoid):
            return self.g(z)
        else:
            return z

    #Softmax layer after output 
    def softmax(self, output):
        dist = []
        denominator = sum(math.e**i for i in output)
        for elem in output:
            dist.append((math.e**elem)/denominator)
        return dist

    #The hidden + output layers of the network, minus the softmax layer
    def completeNetwork(self, w1, w2, x):
        hiddenLayer = []
        outputLayer = []
        for wi in w1:
            hiddenLayer.append(self.neuralUnit(wi,x))
        for wj in w2:
            outputLayer.append(self.neuralUnit(wj,hiddenLayer,sigmoid=False))
        self.finalLayer = self.softmax(outputLayer)
        return self.finalLayer
    

    #Get the largest indices here
    def maxIndex(self, output):
        max = 0
        for i in range(len(output)):
            if output[max] < output[i]:
                max = i
        return max
    
    #Classifyier a given input by returning the max index i.e the index that had the highest probability
    def classifier(self, w1, w2, x):
        return self.maxIndex(self.completeNetwork(w1,w2,x))
    
    #Calculate the error rate of the predictions
    def errorRate(self,predict,y):
        error = 0
        for i in range(len(y)):
            if predict[i] != y[i]:
                error += 1
        return (error/len(y))
    

    #Feed forward the values of x to the hidden layer and then to the output layer
    def forwardProp(self,w1,w2,x,y):
        startTime = time.time()
        self.predictions = []
        for image in x:
            self.predictions.append(self.classifier(w1,w2,image))
        endTime = time.time()
        return self.errorRate(self.predictions,y), endTime-startTime

    #Get a vectorised version of y
    def toCategorical(self,y):
        self.catY = []
        dim  = 10
        for yi in y:
            oneShot = zeroVec(dim)
            oneShot[yi] = 1
            self.catY.append(oneShot)
        return self.catY

    #Get the 10 outputs from the output layer
    def getOutputLayer(self,x, w1, w2):
        self.hws = []
        for image in x:
            self.hws.append(self.completeNetwork(w1,w2,image))
        return self.hws

    #Generalised loss function implementation
    def MLE(self, y):
        m = len(y)
        dim = 10
        sum = 0
        for i in range(m):
            hwi = self.hws[i]
            yi = self.catY[i]
            tempSum = 0
            for j in range(dim):
                tempSum += (yi[j]*math.log(hwi[j]) + (1-yi[j])*math.log(1-hwi[j]))*-1
            sum += tempSum
        return sum/m

    #Cross Entropy Loss Function 
    def costFunction(self, y):
        m = len(y)
        J = 0
        for i in range(m):
            hwi = self.hws[i]
            tempSum = 0
            tempSum = -math.log(hwi[y[i]])
            J += tempSum
        return J/m
      



x = ftol("ps5_data.csv") #5000 values of x, each of dim(400)
y = labelPreprocessor(ftol("ps5_data-labels.csv")) # 5000 labels
w1 = ftol("ps5_theta1.csv") # 25 values in w1, each value of dim(401), the first term is the bias
w2 = ftol("ps5_theta2.csv") # 10 values in w2, each value of dim(26), the first term is the bias

net = NeuralNetwork()

error, timeTaken = net.forwardProp(w1,w2,x,y)

hws = net.getOutputLayer(x,w1,w2)
catY = net.toCategorical(y)
mle = net.MLE(y)
cost = net.costFunction(y)

print("####### Non Numpy Implementation ###########\n")
print("The error rate is : ", error) #Reported error is 0.0248
print("The MLE cost function is : ",mle) #Reported MLE Loss Function value is 0.15284346245189523
print("The cross-entropy loss function is : ",cost) #Reported MLE Loss Function value is 0.0868885603747501
print("The time taken is ", timeTaken,"\n") #Reported time taken is approximately 4.167617321014404 (varies on each run)
print("####### End of Non Numpy Implementation ###########\n")

########################################################## END OF IMPLEMENTATION WITHOUT NUMPY ###########################################################


'''

Here's the Neural Net implementation using Numpy.

'''

class NeuralNumpyNetwork():

    def __init__(self):
        pass

    def g(self, z):
        return 1/(1+np.exp(-z))
    
    def softmax(self, ax):
        return (np.exp(ax)/np.sum(np.exp(ax), axis=0))

    def completeNetwork(self, w1, w2, x):
        hiddenLayer = np.apply_along_axis(self.g,0,np.matmul(x,w1))
        hiddenLayer = np.insert(hiddenLayer, 0, 1.0, axis=1)
        self.outputLayer = np.apply_along_axis(self.softmax, 1,np.matmul(hiddenLayer,w2))
        return self.outputLayer

    def getMax(self, output):
        return np.argmax(output, axis=0)
    
    def classifier(self):
        t1 = time.time()
        self.pred = np.apply_along_axis(self.getMax,1,self.outputLayer)
        t2 = time.time()
        return self.pred, t2-t1

    def errorRate(self,y):
        equal = np.sum(y == self.pred)
        errors = (len(y)-equal)/len(y)
        return errors

############################################################### END OF NUMPY IMPLEMENTATION ###########################################################


print("############ Numpy Implementation ##############\n")

x1 = np.genfromtxt('ps5_data.csv',delimiter=',')
x1 = np.insert(x1, 0, 1.0, axis=1)
y1 = np.genfromtxt('ps5_data-labels.csv',delimiter=',')
y1 = y1-1
w1Prime = np.genfromtxt('ps5_theta1.csv',delimiter=',')
w1Prime = w1Prime.T
w2Prime = np.genfromtxt('ps5_theta2.csv',delimiter=',')
w2Prime = w2Prime.T

net = NeuralNumpyNetwork()

outputLayer = net.completeNetwork(w1Prime, w2Prime,x1)
pred, time = net.classifier()

error1 = net.errorRate(y1)
print("The error rate is : ", error1) #Reported error is 0.0248
print("The time taken is ", time,"\n") #Reported time taken is approximately 0.0251 (varies on each run)

print("The NeuralNumpyNetwork is ",timeTaken/time," faster than the regular Neural Network\n")
print("######################## END ########################################")

