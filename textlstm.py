"""
        / ___|  __ _ _   _  __ _ _ __ | |_ __ _ _ __   |  _ \  __ _ ___
        \___ \ / _` | | | |/ _` | '_ \| __/ _` | '_ \  | | | |/ _` / __|
         ___) | (_| | |_| | (_| | | | | || (_| | | | | | |_| | (_| \__ \
        |____/ \__,_|\__, |\__,_|_| |_|\__\__,_|_| |_| |____/ \__,_|___/
                     |___/

"""

import h5py
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


filename = "sample.txt"
raw_text=open(filename).read()
raw_text=raw_text.lower()

chars=sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))

#data cleaning going on...

n_chars=len(raw_text)
n_vocab=len(chars)
print ("Total Characters: "), n_chars
print ("Total Vocab: "), n_vocab

seq_length=100
dataX=[]
dataY=[]
for i in range(0,n_chars - seq_length,1):
    seq_in=raw_text[i:i+seq_length]
    seq_out=raw_text[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns=len(dataX)
print ("Total Patterns: "),n_patterns


X=numpy.reshape(dataX, (n_patterns, seq_length,1))
X=X/float(n_vocab)
y=np_utils.to_categorical(dataY)

#Actual Model starts here.......
model=Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')



#There is no test dataset. We are modeling the entire training dataset to learn the probability of each character in a sequence.


filepath="weights-improvement-{epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]


model.fit(X,y,epochs=20,batch_size=128,callbacks=callbacks_list)




