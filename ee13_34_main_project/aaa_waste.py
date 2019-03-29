import numpy as np
import random
import soundfile as sf
from python_speech_features import mfcc
import sounddevice as sd

#name = "back"
#b = "z" + name +"_" +str(1)+".wav" 
#data, samplerate = sf.read(b)
#print(data.shape)
#data1 = mfcc(data,samplerate)
#print(data1.shape)
##data = data1.reshape(4043,)
#

#a,b = sf.read("back1.wav")
duration = 2
fs = 8000
data = sd.rec(int(duration*fs), fs, 1)
sd.wait()
#sd.play(data, fs)
#sd.wait()
	
print (len(data), fs)
x= len(data)
p = 25000-x
new_data = np.empty([25000,]) #creating an empty array for new file to be generated from original file
y1 = np.empty([25000,])
#print(p)
for y in range(0, p, int(p/25)):
	for i in range(0, y):
		new_data[i] = y1[i]
	for i in range(y, y+x):	
		new_data[i] = data[i-y]
	for i in range(y+x,25000):
		new_data[i] =  y1[i]
	
#sd.play(new_data, fs)
#sd.wait()
new_data = mfcc(new_data,fs, numcep=13)
data1 = new_data.reshape(4043,)
