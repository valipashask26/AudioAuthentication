import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import os
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

def load_wavs_as_matrices(filepath):
    x, sr = librosa.load(filepath)
    #mfccs=librosa.feature.mfcc(y,sr)
    #return mfccs.T
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = mfccs.ravel()#reshape(mfccs.shape[0] * mfccs.shape[1])
    strs = ''
    for i in range(len(mfccs)):
        strs+=str(mfccs[i])+","
    strs = strs[0:len(strs)-1]    
    return strs

'''
path = 'dataset'
X = []
Y = []

def load_wavs_as_matrices(filepath):
    x, sr = librosa.load(filepath)
    #mfccs=librosa.feature.mfcc(y,sr)
    #return mfccs.T
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = mfccs.ravel()#reshape(mfccs.shape[0] * mfccs.shape[1])
    strs = ''
    for i in range(len(mfccs)):
        strs+=str(mfccs[i])+","
    strs = strs[0:len(strs)-1]    
    return strs
    
data = ''
for i in range(0,4121):
    data+="F"+str(i)+","
data = data[0:len(data)-1]
data+="\n"
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        features = load_wavs_as_matrices(root+"/"+directory[j])
        data+=features+"\n"
        if name == 'forge':
            Y.append(1)
        if name == 'real':
            Y.append(0)

f = open("my.csv", "w")
f.write(data)
f.close()          
Y = np.asarray(Y)
np.save("Y.txt",Y)
'''
Y = np.load("Y.txt.npy")
dataset = pd.read_csv("my.csv")
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
#np.save("xtest.txt",X_test)
#np.save("ytest.txt",y_test)
X_test = np.load("xtest.txt.npy")
y_test = np.load("ytest.txt.npy")
print(y_train)
print(y_test)
'''
rfc = GaussianMixture(n_components=2, random_state=42)
rfc.fit(X, Y)
prediction_data = rfc.predict(X_test)
print(prediction_data)
accuracy = accuracy_score(y_test,prediction_data)*100
print(accuracy)
with open('model.txt', 'wb') as file:
    pickle.dump(rfc, file)
file.close()
'''
with open('model.txt', 'rb') as file:
    rfc = pickle.load(file)
file.close()
prediction_data = rfc.predict(X_test)
print(prediction_data)
accuracy = accuracy_score(y_test,prediction_data)*100
print(accuracy)
data = ''
for i in range(0,4121):
    data+="F"+str(i)+","
data = data[0:len(data)-1]
data+="\n"

data+=load_wavs_as_matrices("dataset/real/200003_real.wav")
f = open("test.csv", "w")
f.write(data)
f.close()

test = pd.read_csv("test.csv")
test.fillna(0, inplace = True)
test = test.values
test = test[:,0:(dataset.shape[1]-1)]
#scaler = MinMaxScaler()
test = scaler.transform(test)
test = pca.transform(test)
print(rfc.predict(test))
'''

X = np.asarray(X)
Y = np.asarray(Y)


print(X.shape)
print(Y.shape)
print(X)
#pca = PCA(n_components = 100)
#X = pca.fit_transform(X)
#print(X.shape)          
'''
