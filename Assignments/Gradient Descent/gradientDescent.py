'''
Abdullah Zameek - Assignment 02 Gradient Descent

'''

######################### MATH AND HELPER FUNCTIONS START HERE ######################################

def calcMean(valList):
    tot = 0
    for val in valList:
        tot+=val
    return tot/len(valList)

def calcstd(valList):
    var =0
    mean = calcMean(valList)
    for val in valList:
        var += (val-mean)**2
    return ((var/len(valList))**0.5)

def ftol(filename):
    stream = open(filename, "r")
    returnList = []
    for elem in stream:
        elem = elem.split()
        returnList.append(elem)
    return returnList

###############################   END OF MATH AND OTHER FUNCTIONS ###########################################

def normalizeData(textFile):
    size = []
    noBr = []
    price = []
    housingData = ftol(textFile)
    for line in housingData:
        line = ''.join(line)
        line = line.split(',')
        size.append(int(line[0]))
        noBr.append(int(line[1]))
        price.append(int(line[2]))

    impVales = [calcMean(size),calcstd(size),calcMean(noBr),calcstd(noBr),calcMean(price),calcstd(price)]
    
    fileList = []
    for i in range(len(size)):
        size[i] = (size[i]-impVales[0])/impVales[1]
        noBr[i] = (noBr[i]-impVales[2])/impVales[3]
        price[i] = (price[i]-impVales[4])/impVales[5]
        fileList.append([str(size[i])+","+str(noBr[i])+","+str(price[i])+"\n"])
    
    normalized = open("normalized.txt",'w')
    for item in fileList:
        item = ''.join(item)
        normalized.write(item)
    normalized.close()

class gradientDescent():
    
    def __init__(self,learningRate):
        self.learningRate = learningRate
    

def main():
    a = [1,2,3]
    print(calcMean(a))
    print(calcstd(a))
    normalizeData("housing.txt")

if __name__ == "__main__":
	main()
