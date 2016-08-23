import argparse
import sys
import random
random.seed(123)
sys.path.append('../../')
sys.path.append('../')
import tensorflow as tf
import numpy as np
#from evaluate import strict, loose_macro, loose_micro
from hooks import acc_hook, loss_hook
import modules
from create_prior_knowledge import create_prior
from batcher import Batcher
from tensorflow.python.ops import variable_scope as vs
from sklearn.externals import joblib

BATCH_SIZE = 1000
LABEL_SIZE = 113
INPUT_SIZE = 300
FEATURE_SIZE = 588455

parser = argparse.ArgumentParser()
parser.add_argument("context_length", help="", type=int)
#parser.add_argument("target_length", help="", type=int)
parser.add_argument("lstm_hidden_size", help="", type=int)
parser.add_argument("attention_hidden_size", help="", type=int)
parser.add_argument("keep_prob_context", help="", type=float)
parser.add_argument("keep_prob_target", help="", type=float)
parser.add_argument("model_name", help="", type=str)
args = parser.parse_args()


# MODEL

## placeholders
keep_prob_target = tf.placeholder(tf.float32)
keep_prob_context = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32,[None,LABEL_SIZE])
x_context = [tf.placeholder(tf.float32, [None, INPUT_SIZE]) for _ in range(args.context_length*2+1)]
x_context_left = x_context[:args.context_length]
x_context_right = x_context[args.context_length+1:]
x_target = tf.placeholder(tf.float32, [None, INPUT_SIZE])

## three outputs
### context
with tf.variable_scope("context_left"):
    rnn_out_context_left = modules.bidirectional_lstm(x_context_left,keep_prob_context,INPUT_SIZE,args.lstm_hidden_size,args.context_length)
with tf.variable_scope("context_right"):    
    rnn_out_context_right = modules.bidirectional_lstm(x_context_right,keep_prob_context,INPUT_SIZE,args.lstm_hidden_size,args.context_length)
out_context,attention_context = modules.importance(rnn_out_context_left+rnn_out_context_right,args.lstm_hidden_size*2,args.attention_hidden_size,args.context_length*2)
    
### entity    
out_target  = tf.nn.dropout(x_target,keep_prob_target)

## output
concat = tf.concat(1,[out_context,out_target])

W = tf.Variable(tf.random_uniform([LABEL_SIZE, INPUT_SIZE+args.lstm_hidden_size*2],minval=-0.01, maxval=0.01))
S = tf.constant(create_prior("../../resource/label2id_figer.txt"),dtype=tf.float32)
V = tf.matmul(S,W)

pre_activation = tf.matmul(concat,V,transpose_b=True) # prior
#pre_activation = tf.matmul(concat,W,transpose_b=True) # no prior
output = tf.nn.sigmoid(pre_activation)
# no_prior


## loss,optimizer,init
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pre_activation, y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.initialize_all_variables()


## batcher
print "loading dataset..."
train_dataset = joblib.load("../../data/figer_train")
dev_dataset = joblib.load("../../data/figer_dev")
test_dataset = joblib.load("../../data/figer_test")
print "loading dicts..."
dicts = joblib.load("../../data/dict_figer")
print "obtaining batch..."
train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],BATCH_SIZE,args.context_length,dicts["id2vec"])
dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],10000,args.context_length,dicts["id2vec"]) #2202 10000
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],563,args.context_length,dicts["id2vec"]) #8885 563

# saver
saver = tf.train.Saver()

# session
sess = tf.Session()
sess.run(init)
#print("restoring...")
#saver.restore(sess, "./mmm10.ckpt")


#Profiling
"""
from line_profiler import LineProfiler
profiler = LineProfiler()
profiler.add_function(train_batcher.next)
profiler.add_function(train_batcher.create_input_output)
profiler.add_function(train_batcher.mean_vec)
profiler.enable()
x_data, y_data = train_batcher.next()
profiler.print_stats()
"""

[x_context_data, x_target_mean_data, y_data, feature] = test_batcher.next()
test_feed = {y:y_data,keep_prob_context:[1],keep_prob_target:[1]}
for i in range(args.context_length*2+1):
    test_feed[x_context[i]] = x_context_data[:,i,:]
test_feed[x_target] = x_target_mean_data


ite = 0
#[x_context_data, x_target_mean_data, y_data] = batcher.next()
#print x_context_data
train_batcher.shuffle()
# TRAINING
l = 0.
for step in range(50000001):
    [x_context_data, x_target_mean_data, y_data, feature] = train_batcher.next()
    feed = {y:y_data,keep_prob_context:[args.keep_prob_context],keep_prob_target:[args.keep_prob_target]}
    for i in range(args.context_length*2+1):
        feed[x_context[i]] = x_context_data[:,i,:]
    feed[x_target] = x_target_mean_data
    sess.run(optimizer,feed_dict = feed)
    l += sess.run(loss,feed_dict = feed) 
    if step % 260 == 0 and step > 1:
        ite += 1
        #print step,l/100.
        #batcher.batch_num = 0
        train_batcher.shuffle()
        lte = sess.run(loss,feed_dict = test_feed)
        print "test loss",step,lte/563.
        print "train loss",l/20000.
        l = 0.0
        print "=== train ==="
        acc_hook(train_batcher,output,x_context,x_target,y,keep_prob_context,keep_prob_target,args.context_length,sess)
        print "=== dev ==="
        acc_hook(dev_batcher,output,x_context,x_target,y,keep_prob_context,keep_prob_target,args.context_length,sess)
        print "=== test ==="
        acc_hook(test_batcher,output,x_context,x_target,y,keep_prob_context,keep_prob_target,args.context_length,sess)
        save_path = saver.save(sess, "./"+args.model_name.strip()+str(ite)+".ckpt")
        print("Model saved in file: ", save_path)
        
    
