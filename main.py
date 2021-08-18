import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string 
import random

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from zipfile import ZipFile

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

with ZipFile ("depression.zip","r") as zip:
    zip.extractall()

df = pd.read_csv("sentiment_tweets3.csv")

def preprocess(x):
    x = x.lower()
    x = x.encode("ascii","ignore").decode()
    x = re.sub("https*\S+"," ",x)
    x = re.sub("@\S+"," ",x)
    x = re.sub("#\S+"," ",x)
    x = re.sub("\'\w+","",x)
    x = re.sub("[%s]" % re.escape(string.punctuation)," ",x)
    x = re.sub("\w*\d+\w*","",x)
    x = re.sub("\s{2,}"," ",x)
    return x

temp = []
data_to_list = df["message to examine"].values.tolist()
for i in range(len(data_to_list)):
    temp.append(preprocess(data_to_list[i]))

def tokenize(y):
    for x in y:
        yield(word_tokenize(str(x)))

data_words = list(tokenize(temp))

def detokenize(txt):
    return TreebankWordDetokenizer().detokenize(txt)

final_data = []
for i in range(len(data_words)):
    final_data.append(detokenize(data_words[i]))
final_data = np.array(final_data)
print(final_data[:5])

import pickle
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

labels = df["label (depression result)"]

max_words = 8225
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(final_data)
sequences = tokenizer.texts_to_sequences(final_data)
tweets = pad_sequences(sequences,maxlen=max_len)
with open("tokenizer.pickle","wb") as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
print(tweets)

x_train,x_test,y_train,y_test = train_test_split(tweets,labels,random_state=42)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

def model():
    inputs = tf.keras.Input(shape=(None,),dtype="int32")
    x = layers.Embedding(max_words,128)(inputs)
    x = layers.Bidirectional(layers.LSTM(64,return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(2)(x)
    model = tf.keras.Model(inputs,outputs)
    return model

model = model()
model.summary()

if __name__=="__main__":
    model.compile(tf.keras.optimizers.Adam(),tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    checkpoint = ModelCheckpoint("gru.h5",monitor="val_accuracy",verbose=1,save_best_only=True,save_weights_only=False)
    model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),callbacks=[checkpoint])
    best = tf.keras.models.load_model("gru.h5")
    loss,acc = best.evaluate(x_test,y_test,verbose=2)
    print("\033[1m")
    print("\nAccuracy: {:.2f} %".format(100*acc))
    print("Loss: {:.2f} %".format(100*loss))
    print("\033[0m")