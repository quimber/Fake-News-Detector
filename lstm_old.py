import numpy as np
filename = 'glove.6B.50d.txt'
deatset='sick.txt'
threshold=0.5
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab,embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)

import tensorflow as tf
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

with tf.Session() as sess:
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

from tensorflow.contrib import learn
#init vocab processor
vocab_processor = learn.preprocessing.VocabularyProcessor(15)
#fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
A= numpy.zeros(shape=(50,20,9840))
B=numpy.zeros(shape=(50,20,9840))
score=numpy.zeros(shape=(1,1,9840))
file=open(dataset,'r')
for line1 in file.readlines():
    line=line1[4:]
    if(line[0]==" "):
        continue
    elif(line1[0=='s']):
        score[ : : i]=line1[8]    
    elif(line1[0]=="A"):
        sent_dict=line.lower.split(" ")
        for i in range 20:
            x = np.array(list(vocab_processor.transform(((sent_dict[i])))))
            word_vec=(tf.nn.embedding_lookup(W, x))
            word_vec.reshape(50,1)
            A[ : i : ]=word_vec
    elif(line1[0]=="B"):
        line1=line[4:]
        sent_dict=line1.lower.split(" ")
        for i in range 20:
            x = np.array(list(vocab_processor.transform(((sent_dict[i])))))
            word_vec=(tf.nn.embedding_lookup(W, x))
            word_vec.reshape(50,1)
            B[ : i : ]=word_vec 

T1=tf.placeholder(tf.float32,[None,50,20,1])
T2=tf.placeholder(tf.float32,[None,50,20,1])
cell=tf.contrib.RNNBasicLSTMcell(20)
initial_state=cell.zero_state(32,tf.float32)
output1=tf.nn.dynamic_rnn(cell, T1,initial_state=initial_state,dtype=tf.float32)
output2=tf.nn.dynamic_rnn(cell, T2,initial_state=initial_state,dtype=tf.float32)
score_=np.matmul(output1,output2)
score_place=tf.placeholder(tf.float32,[None,1])
error=tf.losses.mean_squared_error(score_place,score)
optimizer=tf.train.AdamOptimizer(0.3).minimize(error)

u=0
with tf.Session() as Sess:
    Sess.run(global_variables_initializer)
    for u in range(9840):
        Sess.run(optimizer,{T1:A[50,20,u:u+32],T2:B[50,20,u:u+32],score_place:score[i:i+32]})
        u+=32

