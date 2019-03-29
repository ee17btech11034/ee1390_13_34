import wave
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
import os
import bluetooth
from aaa_record import record
print "Searching for devices..."
print ""
nearby_devices = bluetooth.discover_devices()
num = 0
print "Select your device by entering its coresponding number..."
for i in nearby_devices:
	num+=1
	print num , ": " , bluetooth.lookup_name( i )

selection = input("> ") - 1
print "You have selected", bluetooth.lookup_name(nearby_devices[selection])
bd_addr = nearby_devices[selection]

port = 1
factor = True

class App:
	sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM)
	def __init__(self):
		self.sock.connect((bd_addr, port))
		
	def sending(self,x):
		self.sock.send(x)
	
inst = App()
while factor:
	x = raw_input("Press r to record: ")
	if x=='r':
		value = record()
		inst.sending(str(value))
	elif x=='q':
		factor = False
		print("Terminating Program");

			
