import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Read the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
img = [map(int,a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))

######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.

def normalize(U):
    U = np.array(U)
    square_sum = np.sum(U*U, axis=0)
    norm = np.repeat((square_sum**0.5).reshape(1,square_sum.size),U.shape[0], axis=0)
    return U/norm
    
    

	#return U / LA.norm(U) 

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: 1. You need a matplotlib.pyplot.show() at the end to display all the figures.


first_face = np.reshape(faces[0],(64,64),order='F')

image_count+=1
plt.figure(image_count)
plt.title('First_face')
plt.imshow(first_face,cmap=plt.cm.gray)


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
# Note: There are two ways to order the elements in an array: 
#       column-major order and row-major order. In the np.reshape(), 
#       you can set the order by ``order='C'" for row-major, 
#       or by ``order='F'" for column major. 

#### Your Code Here ####
np.random.seed(1)
idx = np.random.choice(range(faces.shape[0]))
f = faces[idx].reshape((64, 64), order='F')
plt.figure()
plt.title('face No.{}'.format(idx))
plt.imshow(f,cmap=plt.cm.gray)

########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####
average = np.mean(faces, axis=0)
average_face = average.reshape((64, 64), order='F')
plt.figure()
plt.title('Average face')
plt.imshow(average_face,cmap=plt.cm.gray)
plt.title('centralized face')
plt.imshow(f-average_face,cmap=plt.cm.gray)

######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####
ave_faces = np.repeat(average.reshape(1, average.size), faces.shape[0], axis=0)
A = faces - ave_faces
plt.figure()
plt.title('face centralized')
plt.imshow(A[0,:].reshape((64,64), order='F'),cmap=plt.cm.gray)


######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####
A = np.matrix(A)
L = A*A.transpose()
eig_vals, eig_vecs = LA.eig(L)
eig_vals_sorted = np.sort(-eig_vals)
eig_vecs_sorted = eig_vecs[:, (-eig_vals).argsort()]
print("eigenvalues", eig_vals_sorted)



########## Display the first 16 principal components ##################

#### Your Code Here ####
Z = A.transpose()*eig_vecs_sorted
U = normalize(Z)
print (U.shape)
print (U[:16])
eigenfaces = U.reshape((64,64,400), order='F')
plt.figure()
plt.title('eigenface')
for i in range(16):
   plt.subplot(4,4,i+1)
   plt.imshow(eigenfaces[:,:,i], cmap=plt.cm.gray)
   plt.axis('off')
a = np.sum(eigenfaces[:,:,:400], axis=2)
b = (a-a.min())/(a.max()-a.min())*255
b = np.array(b,dtype='int8')
#plt.imshow(eigenfaces[:,:,0], cmap=plt.cm.gray)


########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####
#omega = np.matrix(eig_vals_sorted.reshape(eig_vals_sorted.size,1))
omega = np.matrix(U[:,:2]).transpose()*np.matrix(A[2,:]).transpose()
rebuild = (np.matrix(U[:,:2])*omega).transpose() + average
ax = plt.figure()
plt.title('rebuild')
plt.imshow(rebuild.reshape((64,64), order='F'), cmap=plt.cm.gray)
plt.show()
ax.savefig("first_face_recons.pdf")

########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####
plt.figure()
plt.title('rebuild')
j = 1
for i in [1, 5, 10, 25, 50, 100, 200, 300, 400]:
    omega = np.matrix(U[:,:i-1]).transpose()*np.matrix(A[99,:]).transpose()
    rebuild = (np.matrix(U[:,:i-1])*omega).transpose() + average
    plt.subplot(3,3,j)
    plt.imshow(rebuild.reshape((64,64), order='F'), cmap=plt.cm.gray)
    j += 1
plt.show()
#plt.imshow(rebuild.reshape((64,64), order='F'), cmap=plt.cm.gray)




######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
ax = plt.figure()
plt.title("The proportion of variance of all the PCs")
plt.plot(-eig_vals_sorted/np.sum(-eig_vals_sorted))
ax.savefig("pca.pdf")