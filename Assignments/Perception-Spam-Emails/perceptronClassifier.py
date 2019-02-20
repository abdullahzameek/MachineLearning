'''
Abdullah Zameek

16th February 2019 - Perceptron Spam Email Classifier


Helper Math and Other Support functions.
Since numpy and other external libraries are forbidden, the following helper functions will
help with the linear algebra and other mathematical operations associated with the Perceptron, 
as well as with some file manipulation
'''
######################### MATH AND HELPER FUNCTIONS START HERE ######################################

def vectorAdd(x,y):
    sum = []
    if(len(x) != len(y)):
        print("The lists have to be of equal length!")
        return None
    else:
        sum = [i+j for i,j in zip(x,y)]
        return sum

def dotProduct(x,y):
    if(len(x) != len(y)):
        print("The lists have to be of equal length!")
        return None
    else:
        return sum(i*j for i,j in zip(x,y))

def scalarMultiply(scalar,vect):
    return [scalar*i for i in vect]


def zeroVec(numZeroes):
    zeroVec = []
    zeroVec = [0]*numZeroes
    return zeroVec

def ftol(filename):
    stream = open(filename, "r")
    returnList = []
    for elem in stream:
        elem = elem.split()
        returnList.append(elem)
    return returnList

###############################   END OF MATH AND OTHER FUNCTIONS ###########################################

class Perceptron():

    def __init__(self, totalSize, trainingSize,validationSize,trainingOrigin, featureSize):
        self.totalSize = totalSize
        self.trainingSize = trainingSize
        self.trainingOrigin = trainingOrigin
        self.featureSize = featureSize
        self.validationSize = validationSize

    def split(self):
        #This function takes the input file and depending on the number of lines that are required for the 
        #training and validation files, the function takes in the training size and outputs two files accordingly

        rawDataFile = open(self.trainingOrigin, "r")
        self.trainingFile = open("training.txt", "w")
        self.validationFile = open("validation.txt", "w")
        self.rawData = rawDataFile.readlines()

        for i in range(self.validationSize):
            self.validationFile.write(self.rawData[i])
        self.validationFile.close()

        for j in range(self.validationSize,self.validationSize+self.trainingSize):
            self.trainingFile.write(self.rawData[j])
        self.trainingFile.close()


    def words(self, data):
        repeatsDict = {}
        self.features = []
        for elem in data:
            elem = elem.split()
            for word in set(elem[1:]):
                if word not in repeatsDict.keys():
                    repeatsDict[word] = 1
                else:
                    repeatsDict[word] += 1
        for word in repeatsDict.keys():
            if repeatsDict[word] >= self.featureSize:
                self.features.append(word)
        return self.features
    
    def feature_vector(self, email):
        featureVec = []
        # email = email.split()
        email = email[1:]
        for word in self.features:
            if word in email:
                featureVec.append(1)
            else:
                featureVec.append(0)
        return featureVec

    def getFeatureVectorsLabels(self,filename):
        emails = ftol(filename)
        allFeatureVectors = []
        allLabels = []
        count = 0
        print("Computing feature vectors..")
        for email in emails:
            allFeatureVectors.append(self.feature_vector(email))
            if(email[0] == '1'):
                allLabels.append(1)
            else:
                allLabels.append(-1)
            count+=1
        print("Features done, labels done")
        return allFeatureVectors, allLabels

    def perceptron_train(self,data):
        self.featureVecs, self.labels = self.getFeatureVectorsLabels(data)
        print("The number of features is ",len(self.features))
        w = zeroVec(len(self.features))
        k = 0
        iterator = 0
        n = len(self.featureVecs)
        separated = False
        passCount = 0
        count = 0
        ans = 0
        print("Here we go...")
        while not separated:
            iterator+=1
            separated = True
            for count in range(n):
                x = self.featureVecs[count]
                y = self.labels[count]
                ans = y*dotProduct(w,x)
                if(ans > 0):
                    pass
                else:
                    k+=1
                    separated = False
                    w = vectorAdd(w,scalarMultiply(y,x))
            print("Parsed through set", iterator, " number of times")
        print("Done")
        return w, k, iterator
    
    def perceptron_error(self,data,w):
        featureVecs, labels = self.getFeatureVectorsLabels(data)
        n = len(featureVecs)
        e = 0

        for count in range(n):
            x = featureVecs[count]
            y = labels[count]
            ans = y*dotProduct(w,x)
            if(ans > 0):
                pass
            else:
                e+=1
        return (e/n*100)


    # def returnMostPositiveNegative(self,w, numToFind):
    #     features = list(self.features)
    #     wVec = list(w)
    #     wVec = wVec.sort()
    #     maxVec = wVec[: numToFind]
    #     minVec = wVec[-numToFind:]

    
    #     return minVec, maxVec0

def main():
    perp = Perceptron(4997,3997, 1000, "spam_train.txt",30)
    perp.split()
    trainingData = open("training.txt","r")
    features = perp.words(trainingData)
    trainingData.close()

    
    w, k, iterator = perp.perceptron_train("training.txt")
    # wCopy = list(w)

    #print("The error rate on the training set is ", perp.perceptron_error("training.txt",w))
    print("The error rate on the validation set is ",perp.perceptron_error("validation.txt",w))
    print("The error rate on the spam_test set is ",perp.perceptron_error("spam_test.txt",w))
    mostNeg, mostPos = perp.returnMostPositiveNegative(w,12)
    print(mostNeg)
    print(mostPos)

if __name__ == "__main__":
	main()
