'''
Abdullah Zameek - Assignment 02 Gradient Descent

'''
from random import shuffle

######################### MATH AND HELPER FUNCTIONS START HERE ######################################

#calculates the mean for a given list
def calcMean(valList):
    tot = 0
    for val in valList:
        tot+=val
    return tot/len(valList)

#calculates the standard deviation for a given list
def calcstd(valList):
    var =0
    mean = calcMean(valList)
    for val in valList:
        var += (val-mean)**2
    return ((var/len(valList))**0.5)

#takes in a file and returns a list of lists
def ftol(filename):
    stream = open(filename, "r")
    returnList = []
    for elem in stream:
        elem = elem.split()
        returnList.append(elem)
    return returnList

#This function creates a vector of n-dimensions and fills it with zeroes
def zeroVec(numZeroes):
    zeroVec = []
    zeroVec = [0]*numZeroes
    return zeroVec

def convertToFloat(l):
    temp =[]
    for line in ftol(l):
        floats = ''.join(line)
        floats = [float(x) for x in floats.strip().split(',')]
        temp.append(floats)
    return temp

#Given two vectors of equal length, this function computes the dot/inner product of the two
def dotProduct(x,y):
    if(len(x) != len(y)):
        print("The lists have to be of equal length!")
        return None
    else:
        return sum(i*j for i,j in zip(x,y))

###############################   END OF MATH AND OTHER FUNCTIONS ###########################################

class GradientDescent:

    def __init__(self,textFile):
        self.textFile = textFile
        
    
    #takes in a textfile and returns a file with normalised data
    def normalizeData(self):
        size = []
        noBr = []
        price = []
        housingData = ftol(self.textFile)
        for line in housingData:
            line = ''.join(line)
            line = line.split(',')
            size.append(int(line[0]))
            noBr.append(int(line[1]))
            price.append(int(line[2]))

        #this list will be used later on too. Pretty important info.
        self.impVales = [calcMean(size),calcstd(size),calcMean(noBr),calcstd(noBr),calcMean(price),calcstd(price)]
        

        fileList = []
        for i in range(len(size)):
            size[i] = (size[i]-self.impVales[0])/self.impVales[1]
            noBr[i] = (noBr[i]-self.impVales[2])/self.impVales[3]
            price[i] = (price[i]-self.impVales[4])/self.impVales[5]
            fileList.append([str(size[i])+","+str(noBr[i])+","+str(price[i])+"\n"])
    
        normalized = open("normalized.txt",'w')
        for item in fileList:
            item = ''.join(item)
            normalized.write(item)
        normalized.close()

    
    def lossFunction(self,textFile,w,m):
        '''
        loss function is given by 
        J(w) = (1/2m) Sigma (y^i -fw(x(i)))^2
        '''
        cost =0
        data = convertToFloat(textFile) #get the normalized data from the textfile.

        for i in range(m):
                cost += (dotProduct([1] + data[i],w))**2
        return cost*(2/m)


        
    def gradientDescent(self, textFile, learningRate=0.05, numIters=100):
        normalized = convertToFloat(textFile)
    
        m = len(normalized) #number of data points in the set
        n = len(normalized[0]) #number of parameters

        w = zeroVec(n)
        for ite in range(numIters):
            w.append(-1) #This is done to subtract y from the each iteration of the sum
            tempW = w
            #if ite%10 == 9 : print("step: ", ite+1, "loss function is currently: ", self.lossFunction("normalized.txt",w,m))
            for j in range(n):
                totSum = 0
                for i in range(m):
                    x = [1] + normalized[i] #we know that x1=1
                    totSum += dotProduct(x,w)*x[j]
                tempW[j] -= (learningRate/m)*totSum
            w = tempW[:-1]
        return w

    def predictVal(self,x,w):
        
        x[0] = (x[0]-self.impVales[0])/self.impVales[1]
        x[1] = (x[1]-self.impVales[2])/self.impVales[3]
        x = [1]+x
        predict = (dotProduct(x,w)*self.impVales[5])+self.impVales[4]
        return predict

    def stochasticGD(self, textFile, learningRate=0.05, numIters=3):
        normalized = convertToFloat(textFile)

        m = len(normalized)
        n = len(normalized[0])

        w = zeroVec(n)
        for ite in range(numIters):
            w.append(-1) 
            tempW = w
            for i in range(m):
                x = [1] + normalized[i]
                for j in range(n):
                    tempW[j] -= learningRate* dotProduct(x,w) * x[j]
                w = tempW
            # print("step: ", ite+1, "loss function is currently: ", self.lossFunction("normalized.txt",w,m))
            w = tempW[:-1]
            shuffle(normalized)  
        return w   


        
            
def main():
    grad = GradientDescent("housing.txt")
    grad.normalizeData()
    w = grad.gradientDescent("normalized.txt")
    w1 = grad.stochasticGD("normalized.txt")
    print(w)
    print(w1)
    x = [2650,4]
    print("The predicted price for x= [2650,4] is ",grad.predictVal(x,w))

if __name__ == "__main__":
	main()
