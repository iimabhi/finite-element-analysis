import numpy as np
from scipy.linalg import eigh
from scipy.linalg import eig
import math
from matplotlib import pyplot as plt

# num_elems = int(input('num = ')) 

p = float(input('p = '))
A = float(input('A = '))
L = float(input('L = '))
E = float(input('E = '))



def bar(num_elems):
	restrained_dofs = [0,]

	# element mass and stiffness matrices for a bar
	m = (p*A*L)*(np.array([[2,1],[1,2]]) / (6. * num_elems))
	print('m = ',m)
	k = ((E*A)/L)*(np.array([[1,-1],[-1,1]]) * num_elems)
	print('k=',k)

	# construct global mass and stiffness matrices
	M = np.zeros((num_elems+1,num_elems+1))
	K = np.zeros((num_elems+1,num_elems+1))

	# assembly of elements
	for i in range(num_elems):
		M_temp = np.zeros((num_elems+1,num_elems+1))
		K_temp = np.zeros((num_elems+1,num_elems+1))
		M_temp[i:i+2,i:i+2] = m
		K_temp[i:i+2,i:i+2] = k
		M += M_temp
		K += K_temp

	# remove the fixed degrees of freedom
	for dof in restrained_dofs:
		for i in [0,1]:
			M = np.delete(M, dof, axis=i)
			K = np.delete(K, dof, axis=i)
			# print('M = ',M)
			# print( 'K = ',K)

	# eigenvalue problem
	evals, evecs = eigh(K,M)
	eIvals, eIvecs = eig(K,M)
	print('evacs = ',evecs)
	print('evals = ',evals)
	print('eIvals = ',eIvals)


# bar(2)
	frequencies = np.sqrt(evals)
	return M, K, frequencies, evecs

user = int(input('num= '))

results = []
for i in range(1,user):
	M, K, frequencies, evecs = bar(i)
	results.append( (i, frequencies[0,]) )
	print ('Num Elems: {} \tFund. Frequency: {}'.format(i, round(frequencies[0],3)))

# # plot the results
elements = np.array([x[0] for x in results])
frequencies   = np.array([x[1] for x in results])

plt.plot(elements,frequencies, '-')
plt.xlim(elements[0], elements[-1])
plt.ylim(frequencies[0], frequencies[-1])
plt.xlabel('Number of Elements')
plt.ylabel('frequencies')
plt.show()
print(elements[0])