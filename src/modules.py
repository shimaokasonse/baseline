import tensorflow as tf
from tensorflow.models.rnn.rnn import rnn,bidirectional_rnn
from tensorflow.models.rnn.rnn_cell import LSTMCell,GRUCell,BasicRNNCell
import numpy as np
from tensorflow.python.ops import variable_scope as vs
#from pontus import NonlinearInputProjectionWrapper

def ffnn(input_,INPUT_SIZE,HIDDEN_SIZE,OUT_SIZE):
    
    W1 = tf.Variable(tf.random_uniform([INPUT_SIZE,HIDDEN_SIZE],minval=-0.01, maxval=0.01, ))
    b1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))

    W2 = tf.Variable(tf.random_uniform([HIDDEN_SIZE,OUT_SIZE],minval=-0.01, maxval=0.01, ))

    output =  tf.matmul(tf.nn.relu(tf.matmul(input_,W1) + b1),W2)
    
    return output

def perceptron(input_,INPUT_SIZE,OUT_SIZE):
    W1 = tf.Variable(tf.random_uniform([INPUT_SIZE,OUT_SIZE],minval=-0.01, maxval=0.01, ))
    #b1 = tf.Variable(tf.zeros([OUT_SIZE]))
    output = tf.matmul(input_,W1) # + b1
    return output


def ffnn_sigmoid(input_,INPUT_SIZE,HIDDEN_SIZE,OUT_SIZE):
    pre_activation =  ffnn(input_,INPUT_SIZE,HIDDEN_SIZE,OUT_SIZE)
    output = tf.nn.sigmoid(pre_activation)
    return output, pre_activation
    

def logistic_regression(input_,INPUT_SIZE,OUT_SIZE):
    pre_activation = perceptron(input_,INPUT_SIZE,OUT_SIZE)
    output = tf.nn.sigmoid(pre_activation)
    return output , pre_activation

def margin_loss(output,labels,margin = 0.5):

    big_number = tf.constant(100,tf.float32)
    margin = tf.constant(margin,tf.float32)

    temp  = output - labels * big_number

    min_score_p = tf.reduce_min(temp,1) + big_number
    max_score_n = tf.reduce_max(temp,1) 
    
    loss = tf.reduce_sum(tf.nn.relu(margin - min_score_p + max_score_n))
    
    return loss

def bidirectional_lstm(inputs,keep_prob,INPUT_SIZE,HIDDEN_SIZE,SEQ_LENGTH):
    initializer = tf.random_uniform_initializer(-0.01,0.01)
    cell_F = LSTMCell(HIDDEN_SIZE, INPUT_SIZE, initializer=initializer)
    cell_B = LSTMCell(HIDDEN_SIZE, INPUT_SIZE, initializer=initializer)
    inputs_ = [tf.nn.dropout(each,keep_prob) for each in inputs]
    outputs = bidirectional_rnn(cell_F, cell_B, inputs_, initial_state_fw=None, initial_state_bw=None, sequence_length=None,dtype=tf.float32)
    return outputs


def unidirectional_lstm(inputs,keep_prob,INPUT_SIZE,HIDDEN_SIZE,SEQ_LENGTH):
    initializer = tf.random_uniform_initializer(-0.01,0.01)
    cell = LSTMCell(HIDDEN_SIZE, INPUT_SIZE,initializer=initializer)
    inputs_ = [tf.nn.dropout(each,keep_prob) for each in inputs]
    outputs,_ = rnn( cell, inputs_,  initial_state=None,  sequence_length=None,dtype=tf.float32)

    return outputs
    
def non_pad_average(inputs):
    average =  tf.add_n(inputs)
    return average


def importance(inputs,INPUT_SIZE,HIDDEN_SIZE,SEQ_LENGTH):
    with tf.variable_scope("importance"):
        W =  tf.Variable(tf.random_uniform([INPUT_SIZE,HIDDEN_SIZE],minval=-0.001, maxval=0.001, ))
        U =  tf.Variable(tf.random_uniform([HIDDEN_SIZE,1],minval=-0.001, maxval=0.001, ))
        tf.get_variable_scope().reuse_variables()
        temp1 = [tf.nn.tanh(tf.matmul(inputs[i],W)) for i in range(SEQ_LENGTH)]
        temp2 = [tf.matmul(temp1[i],U) for i in range(SEQ_LENGTH)]
        pre_activations = tf.concat(1,temp2)
        weights = tf.split(1, SEQ_LENGTH, tf.nn.softmax(pre_activations))
        weighted_inputs = [tf.mul(inputs[i],weights[i]) for i in range(SEQ_LENGTH)]
        output = tf.add_n(weighted_inputs)
    return output, weights

def attention(condition,inputs,CONDITION_SIZE,INPUT_SIZE,HIDDEN_SIZE,SEQ_LENGTH):
    with tf.variable_scope("importance"):
        W =  tf.Variable(tf.random_uniform([INPUT_SIZE,HIDDEN_SIZE],minval=-0.001, maxval=0.001, ))
        V = tf.Variable(tf.random_uniform([CONDITION_SIZE,HIDDEN_SIZE],minval=-0.001, maxval=0.001, ))
        U =  tf.Variable(tf.random_uniform([HIDDEN_SIZE,1],minval=-0.001, maxval=0.001, ))
        tf.get_variable_scope().reuse_variables()
        temp1 = [tf.nn.tanh(tf.matmul(inputs[i],W) + tf.matmul(condition,V)) for i in range(SEQ_LENGTH)]
        temp2 = [tf.matmul(temp1[i],U) for i in range(SEQ_LENGTH)]
        pre_activations = tf.concat(1,temp2)
        weights = tf.split(1, SEQ_LENGTH, tf.nn.softmax(pre_activations))
        weighted_inputs = [tf.mul(inputs[i],weights[i]) for i in range(SEQ_LENGTH)]
        output = tf.add_n(weighted_inputs)
    return output, weights
