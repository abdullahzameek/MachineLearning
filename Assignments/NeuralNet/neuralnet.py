'''
Abdullah Zameek

8th April 2019 - Introduction to Neural Networks


Helper Math and Other Support functions.
The following helper functions will help with the linear algebra and other mathematical operations associated with the Neural Network, 
as well as with some file manipulation

'''
import math
import time

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
    def completeNetwork(self, w1, w2, x, softmax=True):
        hiddenLayer = []
        outputLayer = []
        for wi in w1:
            hiddenLayer.append(self.neuralUnit(wi,x))
        
        for wj in w2:
            outputLayer.append(self.neuralUnit(wj,hiddenLayer,sigmoid=False))

        return self.softmax(outputLayer)
    

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

    #MLE cost function implementation
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

class NeuralNumpyNetwork():

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
    

               
x = ftol("ps5_data.csv") #5000 values of x, each of dim(400)
y = labelPreprocessor(ftol("ps5_data-labels.csv")) # 5000 labels
w1 = ftol("ps5_theta1.csv") # 25 values in w1, each value of dim(401), the first term is the bias
w2 = ftol("ps5_theta2.csv") # 10 values in w2, each value of dim(26), the first term is the bias

net = NeuralNetwork()

error, timeTaken = net.forwardProp(w1,w2,x,y)

hws = net.getOutputLayer(x,w1,w2)
catY = net.toCategorical(y)
mle = net.MLE(y)

print("The error rate is : ", error) #Reported error is 0.0248
print("The MLE cost function is : ",mle) #Reported MLE Loss Function value is 0.15284346245189523
print("The time taken is ", timeTaken) #Reported time taken is approximately 4.167617321014404 (varies on each run)

