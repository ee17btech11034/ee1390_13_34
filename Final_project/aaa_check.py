import numpy as np
import random
import soundfile as sf
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from aaa_NN_methods import * 

hiddenLayerSize1 = 7
hiddenLayerSize2 = 10
inputLayerSize = 4043
outputLayerSize = 5
alpha = 1

W1 = np.loadtxt('abcdW1.out', delimiter=',')
W2 = np.loadtxt('abcdW2.out', delimiter=',')
W3 = np.loadtxt('abcdW3.out', delimiter=',')
B1 = np.loadtxt('abcdB1.out', delimiter=',')
B2 = np.loadtxt('abcdB2.out', delimiter=',')
B3 = np.loadtxt('abcdB3.out', delimiter=',')

data, samplerate = sf.read("back"+str(100)+".wav")
#print (len(data), samplerate)
x= len(data)
p = 25000-x
new_data = np.empty([25000,])
y1 = np.empty([25000,])
#print(p)
if x != 25000:
	for y in range(0, p, int(p/20)):
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
	#print (len(new_data))
data1 = mfcc(new_data,samplerate)
data = data1.reshape(inputLayerSize,)
a4, a3, a2, z4, z3, z2 = forward(data, W1, W2, W3, B1, B2, B3)
e = normalize(data)
g = np.arange(1, e.shape[0]+1)
#print(a4)
plt.scatter(g, e)
#plt.show()
print(np.mean(e))
print(np.var(e))
if np.argmax(a4) == 0:
	print("Back")
elif np.argmax(a4) == 1:
	print("Forward")
elif np.argmax(a4) == 2:
	print("Left")
elif np.argmax(a4) == 3:
	print("Right")
elif np.argmax(a4) == 4:
	print("Stop")
else:	
	print("ERROR!")
