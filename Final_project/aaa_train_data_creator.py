import numpy as np
import random
import soundfile as sf
from python_speech_features import mfcc

numbers = 2000
already = True
print("started...")
if already:
    #____________________________________________________________________
    y0 = np.empty([numbers,4043])                   # reading all the back.wav files, coverting to mfcc format, adding labels and storing in an array
    for j in range(0,numbers):
        name = "back"
        b = "z" + name +"_" +str(j+1)+".wav" 
        #print b
        data, samplerate = sf.read(b)
        data1 = mfcc(data,samplerate)
        data = data1.reshape(4043,)
        y0[j]=data
    y = np.empty([numbers,5])
    for i in range (0,numbers):                      # manually assigning labels
        y[i][0]=1.0
        y[i][1]=0.0
        y[i][2]=0.0
        y[i][3]=0.0
        y[i][4]=0.0 
    y0l = np.append(y0,y,axis=1)
    np.savetxt('a_y0l.out',y0l,delimiter = ',')
    print("y0l shape {}".format(y0l.shape))
    y0l = 0
    y = 0
    y0 = 0
    #____________________________________________________________________
    y1 = np.empty([numbers,4043])
    for j in range(0,numbers):                       # reading all the forward.wav files, coverting to mfcc format, adding labels and storing in an array
        name = "forward"
        b = "z" + name +"_" +str(j+1)+".wav" 
        data, samplerate = sf.read(b)
        data1 = mfcc(data,samplerate)
        data = data1.reshape(4043,)
        y1[j]=data
    y = np.empty([numbers,5])
    for i in range (0,numbers):                      # manually assigning labels
        y[i][0]=0.0
        y[i][1]=1.0
        y[i][2]=0.0
        y[i][3]=0.0
        y[i][4]=0.0
    y1l = np.append(y1,y,axis=1)
    np.savetxt('a_y1l.out',y1l,delimiter = ',')
    print("y1l shape {}".format(y1l.shape))  
    y1l = 0
    y = 0
    y1 = 0
    #____________________________________________________________________
    y2 = np.empty([numbers,4043])
    for j in range(0,numbers):                        # reading all the left.wav files, coverting to mfcc format, adding labels and storing in an array  
        name = "left"
        b = "z" + name +"_" +str(j+1)+".wav" 
        data, samplerate = sf.read(b)
        data1 = mfcc(data,samplerate)
        data = data1.reshape(4043,)
        y2[j]=data
    y = np.empty([numbers,5])
    for i in range (0,numbers):                        # manually assigning labels
        y[i][0]=0.0
        y[i][1]=0.0
        y[i][2]=1.0
        y[i][3]=0.0
        y[i][4]=0.0
    y2l = np.append(y2,y,axis=1)
    np.savetxt('a_y2l.out',y2l,delimiter = ',')
    print("y2l shape {}".format(y2l.shape))
    y2l = 0
    y = 0
    y2 = 0
    #____________________________________________________________________
    y3 = np.empty([numbers,4043])                      # reading all the right.wav files, coverting to mfcc format, adding labels and storing in an array
    for j in range(0,numbers):
        name = "right"
        b = "z" + name +"_" +str(j+1)+".wav" 
        data, samplerate = sf.read(b)
        data1 = mfcc(data,samplerate)
        data = data1.reshape(4043,)
        y3[j]=data
    y = np.empty([numbers,5])
    for i in range (0,numbers):                        # manually assigning labels
        y[i][0]=0.0
        y[i][1]=0.0
        y[i][2]=0.0
        y[i][3]=1.0
        y[i][4]=0.0
    y3l = np.append(y3,y,axis=1)
    np.savetxt('a_y3l.out',y3l,delimiter = ',')
    print("y3l shape {}".format(y3l.shape))  
    y3l = 0
    y = 0
    y3 = 0
    #____________________________________________________________________
    y4 = np.empty([numbers,4043])                      # reading all the stop.wav files, coverting to mfcc format, adding labels and storing in an array   
    for j in range(0,numbers):
        name = "stop"
        b = "z" + name +"_" +str(j+1)+".wav" 
        data, samplerate = sf.read(b)
        data1 = mfcc(data,samplerate)
        data = data1.reshape(4043,)
        y4[j]=data
    y = np.empty([numbers,5])
    for i in range (0,numbers):                         # manually assigning labels
        y[i][0]=0.0
        y[i][1]=0.0
        y[i][2]=0.0
        y[i][3]=0.0
        y[i][4]=1.0
    y4l = np.append(y4,y,axis=1)
    np.savetxt('a_y4l.out',y4l,delimiter = ',')
    print("y4l shape {}".format(y4l.shape))
    y4l = 0
    y = 0
    y4 = 0
    #____________________________________________________________________
    trains = np.empty([9000,4048])                   # using the first 5500 elements of each word in the train set 
    k=0
    y0l = np.loadtxt("a_y0l.out", delimiter=',')
    for j in range(0,1800):
        trains[j]=y0l[k]
        k=k+1
    y0l = 0
    y1l = np.loadtxt("a_y1l.out", delimiter=',')
    k=0
    for j in range(1800,3600):
        trains[j]=y1l[k]
        k=k+1 
    y1l = 0
    y2l = np.loadtxt("a_y2l.out", delimiter=',')
    k=0
    for j in range(3600,5400):
        trains[j]=y2l[k]
        k=k+1 
    y2l = 0
    y3l = np.loadtxt("a_y3l.out", delimiter=',')
    k=0
    for j in range(5400,7200):
        trains[j]=y3l[k]
        k=k+1 
    y3l = 0
    y4l = np.loadtxt("a_y4l.out", delimiter=',')
    k=0
    for j in range(7200,9000):
        trains[j]=y4l[k]
        k=k+1 
    y4l = 0
    print("trains shape {}".format(trains.shape))
    np.random.shuffle(trains)
    trainX = np.empty([9000,4043])                         # spliting of train set into features and labels          
    trainY = np.empty([9000,5])
    for i in range(0,9000):
        trainX[i]=trains[i][:4043]
    for i in range(0,9000):
        trainY[i]=trains[i][4043:]
    np.savetxt('a_trainX.out',trainX,delimiter = ',')
    np.savetxt('a_trainY.out',trainY,delimiter = ',')

    print("trainX shape {}".format(trainX.shape))
    print("trainY shape {}".format(trainY.shape))
    trainX = 0
    trainY = 0
    #____________________________________________________________________
    tests = np.empty([1000,4048])
    y0l = np.loadtxt("a_y0l.out", delimiter=',')                      # using the last 750 elements of each array in the test set
    k = 0
    for j in range(0,200):
        tests[j]=y0l[k]
        k=k+1
    y0l = 0
    y1l = np.loadtxt("a_y1l.out", delimiter=',')
    k = 0    
    for j in range(200,400):
        tests[j]=y1l[k]
        k=k+1
    y1l = 0
    y2l = np.loadtxt("a_y2l.out", delimiter=',')
    k = 0 
    for j in range(400,600):
        tests[j]=y2l[k]
        k=k+1
    y2l = 0
    y3l = np.loadtxt("a_y3l.out", delimiter=',')
    k = 0 
    for j in range(600,800):
        tests[j]=y3l[k]
        k=k+1
    y3l = 0
    y4l = np.loadtxt("a_y4l.out", delimiter=',') 
    k = 0
    for j in range(800,1000):
        tests[j]=y4l[k]
        k=k+1 
    y4l = 0
    print("tests shape {}".format(tests.shape))
    np.random.shuffle(tests)
    #____________________________________________________________________
    testX = np.empty([1000,4043])                           # spliting of test set into features and labels  
    testY = np.empty([1000,5])
    for i in range(0,1000):
        testX[i]=tests[i][:4043]
    for i in range(0,1000):
        testY[i]=tests[i][4043:]
    np.savetxt('a_testX.out',testX,delimiter = ',')
    np.savetxt('a_testY.out',testY,delimiter = ',')

    print("testX shape {}".format(testX.shape))
    print("testY shape {}".format(testY.shape))
    testX = 0
    testY = 0
    #____________________________________________________________________
else:
   print("loading 1 out of 4 files...")
   trainX = np.loadtxt('a_trainX.out', delimiter=',')
   print("loading 2 out of 4 files...")
   trainY = np.loadtxt('a_trainY.out', delimiter=',')
   print("loading 3 out of 4 files...")
   testX = np.loadtxt('a_testX.out', delimiter=',')
   print("loading 4 out of 4 files...")
   testY = np.loadtxt('a_testY.out', delimiter=',')
   print("files have been loaded...")
   print("started...")

#rep = 1
#for i in range(rep):
#    training_function(trainX.T[...,i:(i+1)*1000], trainY.T[...,i:(i+1)*1000], 'aa_code01', NN, 100)
#    print("{} completed out of {}".format(i+1, rep))
#    sleep(0.01)
#------------------------------------------------------------------------------
#for i in range(1,10):
#    data, samplerate = sf.read("zback_"+str(i)+".wav")
#    data1 = mfcc(data,samplerate)
#    data = data1.reshape(4043,1)
#    hihi = np.array([[1],[0],[0],[0],[0]])
#    training_function(data, hihi, "AA_test01", NN, i)
#    data, samplerate = sf.read("zforward_"+str(i)+".wav")
#    data1 = mfcc(data,samplerate)
#    data = data1.reshape(4043,1)
#    hihi = np.array([[0],[1],[0],[0],[0]])
#    training_function(data, hihi, "AA_test01", NN, i+1)
#    print(i)
#------------------------------------------------------------------------------

print("Done!")
#NN01 = NeuralNetwork(4043, 10, 5,'aa_code01', training=False)
#NN01.forward(testX.T)
#cost = NN01.cost_function(testX.T, testY.T)
#print("cost is {}".format(cost))
#____________________________________________________________________
