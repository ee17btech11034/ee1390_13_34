import os 
import sounddevice as sd
import soundfile as sf
from time import sleep

name = "stop"
fs = 8000
duration = 2

for i in range(1, 81):
	if i%2 == 0:
		print("{}th recording started....".format(i))
	else:
		print("{}th recording started..".format(i))
	recording = sd.rec(int(duration*fs), fs, 1)
	sd.wait()
	sf.write((name+str(i)+".wav"), recording, fs)
	if i%2 == 0:
		print("finish recording!....")
	else:
		print("finish recording!..")

	#sleep(0)

#for i in range(1, 81):
#	#print("recording started....")
#	#recording = sd.rec(int(duration*fs), fs, 1)
#	#sd.wait()
#	#sf.write((name+str(i)+".wav"), recording, fs)
#	#print("finish recording!....")
#	q,r = sf.read(name+str(i)+".wav")
#	sd.play(q)
#	sd.wait()
#	print (name+str(i)+".wav")