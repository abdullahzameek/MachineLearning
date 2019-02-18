'''
Abdullah Zameek

16th February 2019 - Perceptron Spam Email Classifier

'''


'''
Helper Math functions.
Since numpy and other external libraries are forbidden, the following helper functions will
help with the linear algebra and other mathematical operations associated with the Perceptron
'''
################# MATH FUNCTIONS START ######################################

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


############ END OF MATH FUNCTIONS ###########################

class Perceptron():

    def __init__(self, trainingSize, trainingOrigin, featureSize):
        self.trainingSize = trainingSize
        self.trainingOrigin = trainingOrigin
        self.featureSize = featureSize

    def split(self):
        #This function takes the input file and depending on the number of lines that are required for the 
        #training and validation files, the function takes in the training size and outputs two files accordingly

        rawDataFile = open(self.trainingOrigin, "r")
        self.trainingFile = open("trainingSet.txt", "w")
        self.validationFile = open("validationFile.txt", "w")
        self.rawData = rawDataFile.readlines()

        for i in range(self.trainingSize):
            self.trainingFile.write(self.rawData[i])
        self.trainingFile.close()

        for j in range(self.trainingSize,len(self.rawData)):
            self.validationFile.write(self.rawData[j])
        self.validationFile.close()


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
        email = email.split()
        email = email[1:]
        for word in self.features:
            if word in email:
                featureVec.append(1)
            else:
                featureVec.append(0)
        return featureVec

    def perceptron_train(self):
        pass
    
    def perceptron_error(self):
        pass

    
def main():
    perp = Perceptron(4000, "spam_train.txt",26)
    perp.split()
    trainingData = open("trainingSet.txt","r")
    features = perp.words(trainingData)
    print(len(features))
    # trainingtest = open("validationFile.txt", "r")
    # data = trainingtest.readline()
    # print(data)
    # print(perp.feature_vector(data))

if __name__ == "__main__":
	main()
