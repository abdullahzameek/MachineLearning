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

x = [1,1]
y = [7,5,8,5,2,4]
print(vectorAdd(x,y))
print(dotProduct(x,y))
print(scalarMultiply(3,y))