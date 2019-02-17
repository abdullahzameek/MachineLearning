'''
Abdullah Zameek

16th February 2019 - Perceptron Spam Email Classifier

'''

class Perceptron():

    def __init__(self, trainingSize, trainingOrigin):
        self.trainingSize = trainingSize
        self.trainingOrigin = trainingOrigin

    def split(self):
        #This function takes the input file and depending on the number of lines that are required for the 
        #training and validation files, the function takes in the training size and outputs two files accordingly
        
        rawDataFile = open(self.trainingOrigin, "r")
        trainingFile = open("trainingSet.txt", "w")
        validationFile = open("validationFile.txt", "w")
        rawData = rawDataFile.readlines()

        for i in range(self.trainingSize-1):
            trainingFile.write(rawData[i])
        trainingFile.close()

        for j in range(self.trainingSize,len(rawData)-1):
            validationFile.write(rawData[j])
        validationFile.close()
        

def main():
    perp = Perceptron(1000, "spam_train.txt")
    perp.split()

if __name__ == "__main__":
	main()
