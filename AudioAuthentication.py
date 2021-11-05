from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import random
import numpy as np
import pickle
import os
import librosa
from sklearn.mixture import GaussianMixture
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
import librosa.display

main = tkinter.Tk()
main.title("An Automatic Digital Audio Authentication/Forensics System")
main.geometry("1300x1200")

global filename
global classifier
global x, y
global X_train, X_test, y_train, y_test
global pca
global scaler

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def plotAudio(filepath):
    x, sr = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = mfccs.ravel()
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    plt.figure(figsize=(20,5))
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    plt.show()
    

def extractFeature(filepath):
    x, sr = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = mfccs.ravel()
    strs = ''
    for i in range(len(mfccs)):
        strs+=str(mfccs[i])+","
    strs = strs[0:len(strs)-1]
    return strs

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

def featureExtraction():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    global scaler
    text.delete('1.0', END)
    Y = np.load("model/Y.txt.npy")
    dataset = pd.read_csv("model/audio.csv")
    dataset.fillna(0, inplace = True)
    X = dataset.values
    X = X[:,0:(dataset.shape[1]-1)]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    pca = PCA(n_components = 30)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_test = np.load("model/xtest.txt.npy")
    y_test = np.load("model/ytest.txt.npy")
    text.insert(END,"Total Audio files found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train Gaussian Mixture Model : "+str(len(X_train))+"\n")
    text.insert(END,"Total records used to test Gaussian Mixture Model  : "+str(len(X_test))+"\n")
    
def buildModel():
    text.delete('1.0', END)
    global classifier
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            classifier = pickle.load(file)
        file.close()
        prediction_data = classifier.predict(X_test)
        accuracy = accuracy_score(y_test,prediction_data)*100
        text.insert(END,"Gaussian Mixture Model Prediction Accuracy : "+str(accuracy)+"\n")
    else:
        classifier = GaussianMixture(n_components=2, random_state=42)
        classifier.fit(X, Y)
        prediction_data = rfc.predict(X_test)
        accuracy = accuracy_score(y_test,prediction_data)*100
        text.insert(END,"Gaussian Mixture Model Prediction Accuracy : "+str(accuracy)+"\n")

def authenticateAudio():
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="testSample")
    data = ''
    for i in range(0,4121):
        data+="F"+str(i)+","
    data = data[0:len(data)-1]
    data+="\n"

    data+=extractFeature(file)
    f = open("test.csv", "w")
    f.write(data)
    f.close()

    test = pd.read_csv("test.csv")
    test.fillna(0, inplace = True)
    test = test.values
    test = test[:,0:(test.shape[1]-1)]
    #scaler = MinMaxScaler()
    test = scaler.transform(test)
    test = pca.transform(test)
    predict = classifier.predict(test)[0]
    if predict == 0:
        text.insert(END,"Uploaded File contains REAL authenticate Audio\n")
    if predict == 1:
        text.insert(END,"Uploaded File contains FORGE Audio\n")
    plotAudio(file)    
        

def close():
    main.destroy()
    
    
font = ('times', 15, 'bold')
title = Label(main, text='An Automatic Digital Audio Authentication/Forensics System',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Forge/Real Audio Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

processButton = Button(main, text="Feature Extraction", command=featureExtraction)
processButton.place(x=50,y=200)
processButton.config(font=font1)

buildButton = Button(main, text="Load & Build Gaussian Mixture Model", command=buildModel)
buildButton.place(x=50,y=250)
buildButton.config(font=font1)

graphButton = Button(main, text="Audio Authentication", command=authenticateAudio)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
