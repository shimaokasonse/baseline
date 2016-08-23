import argparse
import sys
sys.path.append('../../')
sys.path.append('../')
import tensorflow as tf
import numpy as np
#from evaluate import strict, loose_macro, loose_micro
from hooks import acc_hook, loss_hook
import modules
from batcher import Batcher
from tensorflow.python.ops import variable_scope as vs
from sklearn.externals import joblib

BATCH_SIZE = 1000
LABEL_SIZE = 90
INPUT_SIZE = 300


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
x_target = tf.placeholder(tf.float32, [None, INPUT_SIZE])

## three outputs
### context
with tf.variable_scope("context"):
    rnn_out_context = modules.bidirectional_lstm(x_context,keep_prob_context,INPUT_SIZE,args.lstm_hidden_size,args.context_length*2+1)
    out_context,attention_context = modules.importance(rnn_out_context,args.lstm_hidden_size*2,args.attention_hidden_size,args.context_length*2+1)

### entity    
out_target  = tf.nn.dropout(x_target,keep_prob_target)

## output
concat = tf.concat(1,[out_context,out_target])
output,pre_activation = modules.logistic_regression(concat,INPUT_SIZE+args.lstm_hidden_size*2,LABEL_SIZE)

## loss,optimizer,init
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pre_activation, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.initialize_all_variables()


## batcher


print "loading dicts..."
dicts = joblib.load("../../data/dicts_gillick")
print "obtaining batch..."
test_dataset = joblib.load("../../data/data_test_gillick")
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],8963,args.context_length,dicts["id2vec"])

# saver
saver = tf.train.Saver()

# session
sess = tf.Session()
sess.run(init)
print "restoring..."
saver.restore(sess, args.model_name)





[x_context_data, x_target_mean_data, y_data] = test_batcher.next()
feed = {y:y_data,keep_prob_context:[1],keep_prob_target:[1]}
for i in range(args.context_length*2+1):
    feed[x_context[i]] = x_context_data[:,i,:]
feed[x_target] = x_target_mean_data
scores = sess.run(output, feed_dict = feed)

for score,true_label in zip(scores,y_data):
    for label_id,label_score in enumerate(list(true_label)):
        if label_score > 0:
            print dicts["id2label"][label_id],
    print "\t",
    lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
    print dicts["id2label"][lid],
    for label_id,label_score in enumerate(list(score)):
        if label_score > 0.5:
            if label_id != lid:
                print dicts["id2label"][label_id],
    print   
