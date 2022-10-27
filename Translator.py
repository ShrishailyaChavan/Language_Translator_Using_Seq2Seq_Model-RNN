#!/usr/bin/env python
# coding: utf-8

# In[16]:


#French to english translator


# In[4]:


import string
import re
from numpy import array, argmax, random, take
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 200)


# In[5]:


data_path = 'eng-fra.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read()
lines


# In[6]:



def to_lines(text):
    sents = text.strip().split('\n') #we want to divide the lines on the basis of new line connector
    sents = [i.split('\t') for i in sents] #we wan to split the data on the basis of \t
    return sents


# In[7]:


fra_eng = to_lines(lines)
fra_eng[:5]


# In[8]:


fra_eng = array(fra_eng)
fra_eng[:5]


# In[9]:


fra_eng.shape


# In[10]:


fra_eng = fra_eng[:20000,:]


# #                             **DATA CLEANING**

# In[11]:


fra_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,0]]
fra_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,1]]

fra_eng[:5]


# In[12]:


#convert text to coverage
for i in range(len(fra_eng)):
    fra_eng[i,0] = fra_eng[i,0].lower()
    fra_eng[i,1] = fra_eng[i,1].lower()
    
fra_eng


# In[13]:


#TEXT TO SEQUENCE CONVERSION (WORD TO INDEX MAPPING)


# In[14]:


# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(fra_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1 #here we are startng with one and zero position we have stored for padding

eng_length = 8 #this means that maxwords in a sentence are 8
print('English Vocabulary Size: %d' % eng_vocab_size)


# In[15]:


#prepare french tokenizer
fra_tokenizer = tokenization(fra_eng[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1

fra_length = 8
print('french Vocabulary Size: %d' % fra_vocab_size)


# In[16]:


#using texts_to_sequences it converts sentences to sequence of numbers
#using pad_sequences we are using length which tells how many word we want in a single sentence


# In[17]:


#encode and pad sequences,padding to a maxium sentence length as mentioned above
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences 
    seq = tokenizer.texts_to_sequences(lines)
    #pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


# Now we will encode the sentences. First of all we will encode French Sentences as the input Sentences and Further English Sentences will be the target sequences. 
# This needs to be done on both training as well as testing datasets.

# In[18]:


from sklearn.model_selection import train_test_split

# split data into train and test set
train, test = train_test_split(fra_eng, test_size=0.2, random_state = 12)


# In[19]:


#Prepare the training data 
trainX = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])


# In[20]:


#prepare the validation  data
testX = encode_sequences(fra_tokenizer, fra_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


# # Define our Seq2Seq model architecture:

# In[21]:


def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True)) #work as encoder
    model.add(LSTM(units)) #works as encoder, LSTM  aslo works here to define how many neurons we want over here
    #here we are not writing any output parameters as these layers are acting as processing inputs
    model.add(RepeatVector(out_timesteps)) # repeating the input on the basis of outsteps where are our outsteps are 8
    model.add(LSTM(units, return_sequences=True)) #act as decoder
    model.add(Dense(out_vocab, activation='softmax')) #act as decoder, here no of neurons depends on number of classes
    return model


# In[108]:


#in_vocab means input vocabulary on which sentence we want to translate
#out_vocab is output
#in_timesteps = max length of words that we decided for input vocabulary as we have decided that every sentence must have 8 words
#units = neurons


# In[22]:


#Here we will be using the RMSprep optimizer for this model as it is the best fit when working with recurrent neural networks.

#model compilation

model = define_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 64)
#here our input vocab is stored in fra_vocab_size and eng_vocab_size
#here 512 is number of neurons
rms = optimizers.RMSprop(learning_rate = 0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
#as deep learning gives the output in one hot encoding using loss as sparse_categorical_crossentropy avoids that one hot encoding step


# In[23]:


history = model.fit(trainX, trainY.reshape(train.shape[0], trainY.shape[1], 1),
                   epochs=10, batch_size=64, validation_split = 0)


# In[24]:


preds = model.predict(testX.reshape((testX.shape[0],testX.shape[1])))

#These predictions denote the sequences of integers and we need to convert these integers to their respective words


# In[25]:


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


# In[27]:


preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t) 

    preds_text.append(' '.join(temp))
                    
        


# In[216]:


import pandas as pd
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})


# In[28]:


pred_df.sample(15)


# In[ ]:




