'''
Abdullah Zameek

16th February 2019 - Perceptron Spam Email Classifier


Helper Math and Other Support functions.
Since numpy and other external libraries are forbidden, the following helper functions will
help with the linear algebra and other mathematical operations associated with the Perceptron, 
as well as with some file manipulation
'''
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

#This function converts a file to a list of lists. Used to access the emails with ease
def ftol(filename):
    stream = open(filename, "r")
    returnList = []
    for elem in stream:
        elem = elem.split()
        returnList.append(elem)
    return returnList

###############################   END OF MATH AND OTHER FUNCTIONS ###########################################

class Perceptron():
    
    #Initial the class with the total number of emails and other essential values to split the data.
    def __init__(self, totalSize, trainingSize,validationSize,trainingOrigin, featureSize, iterLimit):
        self.totalSize = totalSize
        self.trainingSize = trainingSize
        self.trainingOrigin = trainingOrigin
        self.featureSize = featureSize
        self.validationSize = validationSize
        self.iterLimit = iterLimit

    def split(self):
        #This function takes the input file and depending on the number of lines that are required for the 
        #training and validation files, the function takes in the training size and outputs two files accordingly

        rawDataFile = open(self.trainingOrigin, "r")
        self.trainingFile = open("training.txt", "w")
        self.validationFile = open("validation.txt", "w")
        self.rawData = rawDataFile.readlines()


        #create the validation file
        for i in range(self.validationSize):
            self.validationFile.write(self.rawData[i])
        self.validationFile.close()

        #create the training data set
        for j in range(self.validationSize,self.validationSize+self.trainingSize):
            self.trainingFile.write(self.rawData[j])
        self.trainingFile.close()

    #generate the bag of words that contains the "bag of features"
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
    
    #creates a single feature vector
    def feature_vector(self, email):
        featureVec = []
        email = email[1:] #This is get rid of the label 0 or 1
        for word in self.features:
            if word in email:
                featureVec.append(1)
            else:
                featureVec.append(0)
        return featureVec

    #returns all the feature vectors and labels. Done for convenience's sake. 
    def getFeatureVectorsLabels(self,filename):
        emails = ftol(filename)
        allFeatureVectors = []
        allLabels = []
        count = 0
        print("Computing feature vectors..")
        for email in emails:
            allFeatureVectors.append(self.feature_vector(email)) #add individual feature vector to a list of feature vectors
            if(email[0] == '1'): #append the label(y) to a list of labels
                allLabels.append(1)
            else:
                allLabels.append(-1)
            count+=1
        print("Features done, labels done")
        return allFeatureVectors, allLabels

    
    def perceptron_train(self,data):
        self.featureVecs, self.labels = self.getFeatureVectorsLabels(data) #get all the labels and feature vectors
        print("The number of features is ",len(self.features))
        w = zeroVec(len(self.features)) #create the weights vector and initialise it with zeroes
        k = 0 #Number of mistakes
        iterator = 0 #Number of parses/iterations
        n = len(self.featureVecs) #number of feature vectors
        separated = False #This boolean is used as part of the exit condition for the loop
        count = 0 
        ans = 0 #count and ans are used as local variables in the while loop
        print("Here we go...")

        '''
        The loop was implemented as follows:
        "separated" is initially set to false so as to enter the loop.
        Before it enters the for loop, it is set to true, i.e before it loops 
        over the feature vectors in a single pass.
        If the program flow goes into Condition B (marked below) then the
        variable is set to False, and the algorithm executes another pass at the end of the loop.
        If it never enters Condition B, then that means we have successfully found a linearly separating
        hyperplane and we exit with that value of w. The rest of the algorithm is trivial implementation 
        of the perceptron algorithm described in class
        '''
        while not separated and iterator < self.iterLimit: 
            iterator+=1
            separated = True
            for count in range(n):
                x = self.featureVecs[count]
                y = self.labels[count]
                ans = y*dotProduct(w,x)
                #Condition A
                if(ans > 0):
                    pass
                #Condition B
                else:
                    k+=1
                    separated = False
                    w = vectorAdd(w,scalarMultiply(y,x))
            print("Parsed through set", iterator, " times")
        print("Done, a value of w has been found!")
        return w, k, iterator
    
    #Computes the percentage error
    def perceptron_error(self,data,w):
        featureVecs, labels = self.getFeatureVectorsLabels(data)
        n = len(featureVecs)
        e = 0 #error 
        for count in range(n):
            x = featureVecs[count]
            y = labels[count]
            ans = y*dotProduct(w,x)
            if(ans > 0):
                pass
            else:
                e+=1
        return (e/n*100)


    #Returns the most positive-weight and negative-weight associated words
    def returnMostPositiveNegative(self,w, numToFind):
        dict1 = dict()
        mostPos = []
        mostNeg = []
        mostPosIndex = []
        mostNegIndex = []
        count = 0

        #create a dictionary that can hold lists as values. This is so that one key can take multiple values.
        for i in range(len(self.features)):
            if w[i] in dict1:
                dict1[w[i]].append(self.features[i])
            else:
                dict1[w[i]] = [self.features[i]]

        #get most positive-weighted words
        for key in sorted(dict1.keys(), reverse=True):
            for elem in dict1[key]:
                mostPos.append(elem)
                mostPosIndex.append(dict1[key])
                count+=1
                if(count == numToFind):
                    count = 0
                    break

        #get most negative-weighted words
        for key in sorted(dict1.keys()):
            for elem in dict1[key]:
                mostNeg.append(elem)
                mostNegIndex.append(dict1[key])
                count+=1
                if(count == numToFind):
                    count = 0
                    break

        return mostPos, mostNeg


def main():
    perp = Perceptron(4997,3997, 1000, "spam_train.txt",26,200)
    perp.split()
    trainingData = open("training.txt","r")
    features = perp.words(trainingData)
    trainingData.close()

    w, k, iterator = perp.perceptron_train("training.txt")
    

    print("The error rate on the training set is ", perp.perceptron_error("training.txt",w))
    print("The total number of mistakes made is:  ",k)
    print("The total number of passes made through the data is: ", iterator)
    print("The error rate on the validation set is ",perp.perceptron_error("validation.txt",w))
    print("The error rate on the spam_test set is ",perp.perceptron_error("spam_test.txt",w))
    mostPos, mostNeg = perp.returnMostPositiveNegative(w,12)
    print(mostNeg)
    print(mostPos)

if __name__ == "__main__":
	main()
