import numpy as np
import random
import soundfile as sf
from python_speech_features import mfcc
from time import sleep

hiddenLayerSize1 = 7
hiddenLayerSize2 = 10
inputLayerSize = 4043
outputLayerSize = 5
alpha = 1
#=============================================================
def relu(x, nOut):
	x = np.array(x)
	x = x.reshape(nOut,1)
	x = x
	x[x<-6] = 0
	x=x.reshape(-1,nOut)
	return x

def relu_prime(x, nOut):
	x = np.array(x)
	x = x.reshape(nOut,1)
	x = x
	x[x>=-6] = 1
	x[x<-6] = 0
	x=x.reshape(-1,nOut)
	return x

def softmax(x, nOut):
	x = np.array(x)
	x = x.reshape(nOut,1)
	x = x
	x = np.exp(x - np.argmax(x))/(np.sum(np.exp(x - np.argmax(x))))
	x=x.reshape(-1,nOut)
	return x

def softmax_prime(x, nOut):
	x = np.array(x)
	x = x.reshape(nOut,1)
	x = x
	d_x = np.ones((nOut, nOut))
	I = np.identity(nOut)
	value = softmax(x, nOut)
	d_x = np.multiply(value, (I - value.T))
	return d_x

def normalize(A):
	mean = np.mean(A)
	A = A - mean
	variance = np.var(A)
	A = A/variance
	return A

#=================================================================
def sigmoid(x, nOut):
    x = np.array(x)
    x = x.reshape(nOut,1)
    x = x
    for  i in range (0,nOut):
        if (x[i][0] < -700):                            # to prevent overflow error, we have manually defined it to be 0, when input is very low
            x[i][0]=0
        else:
            x[i] = 1/(1+np.exp(-x[i]))  
    x=x.reshape(-1,nOut)
    return x

def sigmoid_prime(x, nOut):
    return sigmoid(x, nOut)*(1-sigmoid(x, nOut))
#================================================================
def forward(inputMatrix, W1, W2, W3, B1, B2, B3):
	inputMatrix = inputMatrix.reshape(1, inputLayerSize)
	inputMatrix = normalize(inputMatrix)
	print(np.max(inputMatrix))
	B1 = B1.reshape(1, hiddenLayerSize1)
	B2 = B2.reshape(1, hiddenLayerSize2)
	B3 = B3.reshape(1, outputLayerSize)
	z2 = alpha*np.dot(inputMatrix, W1) + B1
	a2 = sigmoid(z2, hiddenLayerSize1)
	z3 = np.dot(a2, W2) + B2
	a3 = sigmoid(z3, hiddenLayerSize2)
	z4 = np.dot(a3, W3) + B3
	a4 = sigmoid(z4, outputLayerSize)
	return a4, a3, a2, z4, z3, z2, inputMatrix #added input matrix


def Back_propagation(inputMatrix, ouptputMatrix, W1, W2, W3, B1, B2, B3):
	scalar1 = 0.1
	scalar2 = 0.1
	ouptputMatrix = ouptputMatrix.reshape(1, outputLayerSize)
	inputMatrix = normalize(inputMatrix)
	inputMatrix = inputMatrix.reshape(1, inputLayerSize)
	B1 = B1.reshape(1, hiddenLayerSize1)
	B2 = B2.reshape(1, hiddenLayerSize2)
	B3 = B3.reshape(1, outputLayerSize)
	ans4, ans3, ans2, out4, out3, out2 = forward(inputMatrix, W1, W2, W3, B1, B2, B3)
	delta4 = np.multiply(-(ouptputMatrix - ans4), sigmoid_prime(out4, outputLayerSize))  # since I ma using softmax delta = out - ans
	#delta4 = -(ouptputMatrix - ans4)
	dJdW3 = np.dot(ans3.T, delta4)
	delta3 = np.multiply(np.dot(delta4, W3.T), sigmoid_prime(out3, hiddenLayerSize2))
	dJdW2 = np.dot(ans2.T, delta3)
	delta2 = np.multiply(np.dot(delta3, W2.T), sigmoid_prime(out2, hiddenLayerSize1))
	dJdW1 = alpha*np.dot(inputMatrix.T, delta2)
	dJdB1 = delta2
	dJdB2 = delta3
	dJdB3 = delta4
	W1 = W1 - scalar1 * dJdW1
	W2 = W2 - scalar1 * dJdW2
	W3 = W3 - scalar1 * dJdW3
	B1 = B1 - scalar2 * dJdB1
	B2 = B2 - scalar2 * dJdB2
	B3 = B3 - scalar2 * dJdB3
	return W1, W2, W3, B1, B2, B3

check = 0
testc = 1
def train():
	W1 = np.matrix(np.random.rand(inputLayerSize, hiddenLayerSize1))
	W2 = np.matrix(np.random.rand(hiddenLayerSize1, hiddenLayerSize2))
	W3 = np.matrix(np.random.rand(hiddenLayerSize2, outputLayerSize))
	B1 = np.matrix(np.random.rand(1, hiddenLayerSize1))
	B2 = np.matrix(np.random.rand(1, hiddenLayerSize2))
	B3 = np.matrix(np.random.rand(1, outputLayerSize))
	if check:
		trainX = np.random.rand(9000, 4043)
		trainY = np.random.rand(9000, 5)
		testX = np.random.rand(1000, 4043)
		testY = np.random.rand(1000, 5)
		nEpochs = 1
		for j in range(nEpochs):                                # traing the dataset
		    for i in range(trainX.shape[0]):
		        W1, W2, W3, B1, B2, B3 = Back_propagation(trainX[i], trainY[i], W1, W2, W3, B1, B2, B3)
		    print("Epoch {}".format(j))
	else:
		nEpochs = 40
		print("train1")
		print("loading 1 out of 2 files...")
		trainX1 = np.loadtxt('a_trainX.out', delimiter=',')
		print("loading 2 out of 2 files...")
		trainY1 = np.loadtxt('a_trainY.out', delimiter=',')
		print("files have been loaded...")
		for j in range(nEpochs):                                # traing the dataset
		    for i in range(trainX1.shape[0]):
		        W1, W2, W3, B1, B2, B3 = Back_propagation(trainX1[i], trainY1[i], W1, W2, W3, B1, B2, B3)
		    print("1 Epoch {}".format(j))

		np.savetxt('abcdW1.out',W1,delimiter = ',')                # values of W1 and b stored in different files to be used in the raspberry pi
		np.savetxt('abcdW2.out',W2,delimiter = ',')
		np.savetxt('abcdW3.out',W3,delimiter = ',')
		np.savetxt('abcdB1.out',B1,delimiter = ',')
		np.savetxt('abcdB2.out',B2,delimiter = ',')
		np.savetxt('abcdB3.out',B3,delimiter = ',')

		'''trainX1 = 0
		trainY1 = 0

		W1 = np.loadtxt('abcdW1.out', delimiter=',')
		W2 = np.loadtxt('abcdW2.out', delimiter=',')
		W3 = np.loadtxt('abcdW3.out', delimiter=',')
		B1 = np.loadtxt('abcdB1.out', delimiter=',')
		B2 = np.loadtxt('abcdB2.out', delimiter=',')
		B3 = np.loadtxt('abcdB3.out', delimiter=',')
		print("train2")
		print("loading 1 out of 2 files...")
		trainX2 = np.loadtxt('as_trainX2.out', delimiter=',')
		print("loading 2 out of 2 files...")
		trainY2 = np.loadtxt('as_trainY2.out', delimiter=',')
		print("files have been loaded...")

		for j in range(nEpochs):                                # traing the dataset
		    for i in range(trainX2.shape[0]):
		        W1, W2, W3, B1, B2, B3 = Back_propagation(trainX2[i], trainY2[i], W1, W2, W3, B1, B2, B3)
		    print("2 Epoch {}".format(j))

		np.savetxt('abcdW1.out',W1,delimiter = ',')                # values of W1 and b stored in different files to be used in the raspberry pi
		np.savetxt('abcdW2.out',W2,delimiter = ',')
		np.savetxt('abcdW3.out',W3,delimiter = ',')
		np.savetxt('abcdB1.out',B1,delimiter = ',')
		np.savetxt('abcdB2.out',B2,delimiter = ',')
		np.savetxt('abcdB3.out',B3,delimiter = ',')

		trainX2 = 0
		trainY2 = 0'''
def test():
	W1 = np.loadtxt('abcdW1.out', delimiter=',')
	W2 = np.loadtxt('abcdW2.out', delimiter=',')
	W3 = np.loadtxt('abcdW3.out', delimiter=',')
	B1 = np.loadtxt('abcdB1.out', delimiter=',')
	B2 = np.loadtxt('abcdB2.out', delimiter=',')
	B3 = np.loadtxt('abcdB3.out', delimiter=',')
	if testc:
		print("loading 1 out of 2 files...")
		testX = np.loadtxt('a_testX.out', delimiter=',')
		print("loading 2 out of 2 files...")
		testY = np.loadtxt('a_testY.out', delimiter=',')
		print("files have been loaded...")
		correct = 0
		total = len(testX)
		print("started...")
		for i in range(testX.shape[0]):                        # making predictions and calculating accuracy
		    a4, a3, a2, z4, z3, z2 = forward(testX[i], W1, W2, W3, B1, B2, B3)
		    pred = np.argmax(a4)
		    actual = np.argmax(testY[i])
		    print("Prediction: Type {}".format(pred))
		    print("Actual: Type {}\n".format(actual))
		    if pred == actual:
		        correct +=1
		    if actual == 3:
		    	car = testX[i]
	        
		print("Accuracy: {}%".format((correct*1.0)/total * 100))
		
#train()
#test()
