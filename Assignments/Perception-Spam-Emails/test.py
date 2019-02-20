listA = [1,2,5,3,5,2,4,5,2,7,-5,-5,-7,-9,-15,-45,-25,-100,52,5865,42,1,1,0,5,2,4,8,3,2,4,2,7,88,6,2,1,55,2,66,66,44,77,55,222,2,1,5,2,4]

def Nmaxelements(list1, N): 
    final_list = [] 
    for i in range(0, N):  
        max1 = 0
          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
                  
        final_list.append(list1.index(max1)) 
        list1.remove(max1)
    print(len(listA)) 
    print(final_list)

maxElems = Nmaxelements(listA,12)