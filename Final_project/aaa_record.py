import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from aaa_NN_methods import *

WAVE_OUTPUT_FILENAME = "testab.wav"

CHANNELS = 1
RATE = 8000 #sample rate
RECORD_SECONDS = 2
b =WAVE_OUTPUT_FILENAME

def record():
	print("recording statred...")
	recording = sd.rec(int(RECORD_SECONDS*RATE), RATE, CHANNELS)
	sd.wait()
	print("recording done...")

	sf.write(b,recording, RATE)
	data, samplerate = sf.read(b)
	#print(type(data), type(recording))
	#print(data.reshape(len(data),1).shape, recording.shape)
	#print(recording == data)

	x= len(data)
	p = int(25000-x)
	l =0
	tests = np.empty([200,4043])
	new_data = np.empty([25000,])
	y1 = np.empty([25000,])	

	y = int(p/2);

	new_data[0:y-1] = y1[0:y-1]
	new_data[y:x+y-1] = data[0:x-1]
	new_data[x+y:25000] = y1[x+y:25000]

	data1 = mfcc(new_data,samplerate)
	data = data1
	data = data.reshape(4043,)

	hiddenLayerSize1 = 7
	hiddenLayerSize2 = 10
	inputLayerSize = 4043
	outputLayerSize = 5


	W1 = np.loadtxt('abcdW1.out', delimiter=',')
	W2 = np.loadtxt('abcdW2.out', delimiter=',')
	W3 = np.loadtxt('abcdW3.out', delimiter=',')
	B1 = np.loadtxt('abcdB1.out', delimiter=',')
	B2 = np.loadtxt('abcdB2.out', delimiter=',')
	B3 = np.loadtxt('abcdB3.out', delimiter=',')
	 
	a, a3, a2, z4, z3, z2, inputMatrix = forward(data, W1, W2, W3, B1, B2, B3)
	print(a)
	result = np.argmax(a)
	if result == 0:
		print("Back")
	elif result == 1:
		print("Forward")
	elif result == 2:
		print("Left")
	elif result == 3:
		print("Right")
	elif result == 4:
		print("Stop")
	else:
		print("ERROR!")

	return(str(result))

#record()
# to plot graph
#g = np.arange(1, inputLayerSize+1)
#plt.scatter(g, inputMatrix)
#plt.show()