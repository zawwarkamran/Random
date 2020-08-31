from _future_ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']
test_datum = x[-2]
x_train = x[:-2]
y_train = y[:-2]

idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
	'''
	Input: A is a Nxd matrix
		   B is a Mxd matirx
	Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
	i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
	'''
	A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
	B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
	dist = A_norm+B_norm-2*A.dot(B.transpose())
	return dist
 
#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
	'''
	Given a test datum, it returns its prediction based on locally weighted regression

	Input: test_datum is a dx1 test vector
		   x_train is the N_train x d design matrix
		   y_train is the N_train x 1 targets vector
		   tau is the local reweighting parameter
		   lam is the regularization parameter
	output is y_hat the prediction on test_datum
	'''
	#Computing the weights 'a' first
	a = np.zeros_like(y_train)
	B = test_datum.reshape(1,-1) #reshaped test_datum
	A = x_train[:]
	dist_array = l2(A,B).reshape(-1)
	dist_array = dist_array*(-1.0)/(2*(tau**2))
	exp_dist = np.exp(dist_array - np.max(dist_array)*np.ones_like(dist_array))
	a = exp_dist/np.sum(exp_dist)
	diag_a = np.diag(a)
	
	#Computing w to get to the prediction
	Xt_A_X = np.matmul(np.transpose(x_train),np.matmul(diag_a,x_train))
	Xt_A_y = np.matmul(np.transpose(x_train),np.matmul(diag_a,y_train.reshape(-1,1)))
	theta = np.linalg.solve(Xt_A_X,Xt_A_y)
	y_pred = np.matmul(test_datum.reshape(1,-1),theta).reshape(-1)
	return y_pred[0]
	
	

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
	'''
	Input: x_test is the N_test x d design matrix
		   y_test is the N_test x 1 targets vector        
		   x_train is the N_train x d design matrix
		   y_train is the N_train x 1 targets vector
		   taus is a vector of tau values to evaluate
	output: losses a vector of average losses one for each tau value
	'''
	N_test = x_test.shape[0]
	losses = np.zeros(taus.shape)
	for j,tau in enumerate(taus):
		predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
						for i in range(N_test)])
		losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
	return losses

#to implement
def run_k_fold(x, y, taus, k):
	'''
	Input: x is the N x d design matrix
		   y is the N x 1 targets vector    
		   taus is a vector of tau values to evaluate
		   K in the number of folds
	output is losses a vector of k-fold cross validation losses one for each tau value
	'''
	solution = []
	for i in range(k):
		x_test = x[:int(len(x)/k), :]
		x_train = x[int(len(x)/k):, :]
		y_test = y[:int(len(y)/k)]
		y_train = y[int(len(y)/k):]
#         solution.append(run_on_fold(x_test, y_test, x_train, y_train, taus))
		x = np.append(x_train, x_test, axis=0)
		y = np.append(y_train, y_test, axis=0)
		print(y_test)
	return solution

if _name_ == "_main_":
	# In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
	taus = np.logspace(1.0,3,200)
	losses = run_k_fold(x,y,taus,k=5)
	plt.plot(losses)
	print("min loss = {}".format(losses.min()))