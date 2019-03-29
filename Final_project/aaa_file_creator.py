
import numpy as np
import soundfile as sf
new_data = np.empty([25000,]) #creating an empty array for new file to be generated from original file
y1 = np.empty([25000,])
count = 1
name = "back"
print("started..")
for j in range(1,81):
	b= name+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print (len(data), samplerate)
	x= len(data)
	p = 25000-x
	#print(p)
	for y in range(0, p, int(p/25)):
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
		a = "z" + name +"_" +str(count)+".wav"    #total length becomes 25000
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print (len(new_data))
		count += 1
print("complete 1")
count = 1
name = "forward"
for j in range(1,81):
	b= name+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print (len(data), samplerate)
	x= len(data)
	p = 25000-x
	for y in range(0, p,int(p/25)): 
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
		a = "z" + name +"_" +str(count)+".wav"    #total length becomes 25000
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print (len(new_data))
		count += 1
print("complete 2")
count = 1
name = "left"
for j in range(1,81):
	b= name+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print (len(data), samplerate)
	x= len(data)
	p = 25000-x
	for y in range(0, p, int(p/25)): 
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
		a = "z" + name +"_" +str(count)+".wav"    #total length becomes 25000
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print (len(new_data))
		count += 1
count = 1
print("complete 3")
name = "right"
for j in range(1,81):
	b= name+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print (len(data), samplerate)
	x= len(data)
	p = 25000-x
	for y in range(0, p, int(p/25)): 
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
		a = "z" + name +"_" +str(count)+".wav"    #total length becomes 25000
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print (len(new_data))
		count += 1
print("complete 4")
count = 1
name = "stop"
for j in range(1,81):
	b= name+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print (len(data), samplerate)
	x= len(data)
	p = 25000-x
	for y in range(0, p, int(p/25)): 
		for i in range(0, y):
			new_data[i] = y1[i]
		for i in range(y, y+x):	
			new_data[i] = data[i-y]
		for i in range(y+x,25000):
			new_data[i] =  y1[i]
		a = "z" + name +"_" +str(count)+".wav"    #total length becomes 25000
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print (len(new_data))
		count += 1
print("done")
