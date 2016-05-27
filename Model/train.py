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
            (types,features) = line.strip().split("\t")
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

def sample_batch(X,Y,batch_size):
    permutation = random.sample(range(X.shape[0]),batch_size)
    return X[permutation,:],Y[permutation,:]


X_train = _t(count_vect_x.transform(docs_train))
Y_train = count_vect_y.transform(labels_train).toarray()
X_dev   = _t(count_vect_x.transform(docs_dev))
Y_dev   = count_vect_y.transform(labels_dev).toarray()
X_test  = _t(count_vect_x.transform(docs_test))
Y_test  = count_vect_y.transform(labels_test).toarray()


import tensorflow as tf
from hooks import acc_hook

dim_x = len(count_vect_x.get_feature_names())
dim_y = len(count_vect_y.get_feature_names())

x = tf.placeholder(tf.int64,[None,INDEX_SIZE])
y = tf.placeholder(tf.float32,[None,dim_y])

W = tf.Variable(tf.random_uniform([dim_x,dim_y], -0.001, 0.001))
u = tf.reduce_sum(tf.nn.embedding_lookup(W,x),1)

b = tf.Variable(tf.zeros([dim_y]))

pre_activation = u + b
prediction = tf.nn.sigmoid(pre_activation)

loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pre_activation, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

X_hook, Y_hook = sample_batch(X_train,Y_train,10000)
feed_hook = {x:X_hook, y:Y_hook}
feed_dev = {x:X_dev, y:Y_dev}
for epoch in range(100):
    for i in range(25):
        X_batch, Y_batch = sample_batch(X_train,Y_train,10000)
        feed = {x:X_batch, y:Y_batch}
        sess.run(optimizer, feed_dict = feed)
    print "epoch:",epoch,"train:", sess.run(loss, feed_dict = feed_hook)/10000,"dev:", sess.run(loss, feed_dict = feed_dev)/X_dev.shape[0]
    print "=== TRAIN ==="
    acc_hook(sess.run(prediction,feed_dict={x:X_hook}),Y_hook)
    print "=== DEV ==="
    acc_hook(sess.run(prediction,feed_dict={x:X_dev}),Y_dev)
    print "=== TEST ==="
    acc_hook(sess.run(prediction,feed_dict={x:X_test}),Y_test)
    print
                                                                                
