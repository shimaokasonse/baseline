from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
np.set_printoptions(threshold=np.nan)

CORPUS = "../Data/OntoNotes/"

def load_corpus(file_name):
    docs = []
    labels = []
    with open(file_name) as f:
        for line in f:
            (_, _, _, types,features) = line.strip().split("\t")
            docs.append(features)
            labels.append(types)
    return docs,labels

docs_train, labels_train = load_corpus(CORPUS+"train.txt")
docs_dev, labels_dev     = load_corpus(CORPUS+"dev.txt")
docs_test, labels_test   = load_corpus(CORPUS+"test.txt")

count_vect_x = CountVectorizer(binary=True,tokenizer=lambda s: s.split())
count_vect_y = CountVectorizer(binary=True,tokenizer=lambda s: s.split())

count_vect_x.fit(docs_train   + docs_dev   + docs_test)
count_vect_y.fit(labels_train + labels_dev + labels_test)

INDEX_SIZE = 70

def _t(S):
    num = S.shape[0]
    I = np.zeros((num,INDEX_SIZE),dtype=np.int64)
    for i in range(num):
        index =  S[i][0].nonzero()[1]
        I[i,:len(index)] = index
    return I

def sample_batch(X,data,batch_size):
    permutation = random.sample(range(data.shape[0]),batch_size)
    batch = data[permutation,:]
    return X[batch[:,0],:], np.array([batch[:,1]]).T, np.array([batch[:,2]]).T

from itertools import product
from scipy.sparse import csr_matrix
def transform_batch(Y):
    data = []
    labels = range(Y.shape[1])
    for i in xrange(Y.shape[0]):
        posis = filter(lambda k: Y[i,k], labels)
        negas = filter(lambda k: not Y[i,k], labels)
        for posi,nega in product(posis,negas): 
            data.append([i,posi,nega])
    return np.array(data,np.int64)

X_train = _t(count_vect_x.transform(docs_train))
Y_train = count_vect_y.transform(labels_train).toarray()
X_dev   = _t(count_vect_x.transform(docs_dev))
Y_dev   = count_vect_y.transform(labels_dev).toarray()
X_test  = _t(count_vect_x.transform(docs_test))
Y_test  = count_vect_y.transform(labels_test).toarray()

data_train = transform_batch(Y_train)
data_dev = transform_batch(Y_dev)
data_test = transform_batch(Y_test)

import tensorflow as tf
from hooks import acc_hook

dim_x = len(count_vect_x.get_feature_names())
dim_y = len(count_vect_y.get_feature_names())
dim_h = 50

x = tf.placeholder(tf.int64,[None,INDEX_SIZE])
posi = tf.placeholder(tf.int64,[None,1])
nega = tf.placeholder(tf.int64,[None,1])

H_x = tf.Variable(tf.random_uniform([dim_x,dim_h], -0.001, 0.001))
u_x = tf.reduce_sum(tf.nn.embedding_lookup(H_x,x),1)

H_y = tf.Variable(tf.random_uniform([dim_y,dim_h], -0.001, 0.001))
u_posi = tf.reduce_sum(tf.nn.embedding_lookup(H_y,posi),1)
u_nega = tf.reduce_sum(tf.nn.embedding_lookup(H_y,nega),1)

s_posi = tf.reduce_sum(tf.mul(u_x, u_posi),1)
s_nega = tf.reduce_sum(tf.mul(u_x, u_nega),1)

loss = tf.reduce_mean (tf.nn.relu( 1 - s_posi + s_nega )) #+ tf.square(s_posi)*0.1 + tf.square(s_nega)*0.1)

prediction = tf.matmul(u_x, H_y, transpose_b=True)

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)



for epoch in range(500):
    #data_train.shffle()
    for i in range(10):
        X_batch, Y_batch_P, Y_batch_N = sample_batch(X_train,data_train,1000)
        feed = {x:X_batch, posi:Y_batch_P, nega:Y_batch_N}
        sess.run(optimizer, feed_dict = feed)
        print sess.run(loss,feed_dict = feed)
    print "=== DEV ==="
    acc_hook(sess.run(prediction,feed_dict={x:X_dev}),Y_dev,4)
    print "=== TEST ==="
    acc_hook(sess.run(prediction,feed_dict={x:X_test}),Y_test,4)
        #print "nega",sess.run(s_nega,feed_dict = feed)
        #print "posi",sess.run(s_posi,feed_dict = feed)
                                                                                
