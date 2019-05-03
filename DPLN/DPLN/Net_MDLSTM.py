from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.framework import dtypes
from tensorflow.contrib.rnn import LSTMStateTuple, LayerRNNCell
from tensorflow.python.ops import math_ops, init_ops, array_ops, nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_array_ops
from DPLN.MDLSTM import MDLSTMcell, MDLSTMcell_whileloop, MDLSTM_8direction_output
from DPLN.ProcGeneratorV4 import ProcGanerator 
from tensorflow.python.ops import variable_scope as vs

def MDLSTMNet(input_x, 
              output_size=2,
              hidden_units=[8, 16],
              fc_units=[25], 
              keep_prob=1.0):
    batch_size, img_len, img_len, z_dim, channels=input_x.shape
#     one_layer_param=img_len*img_len*z_dim

    start1 = time.time()
    
    # MDLSTM layer 0
    MDLSTM_layer_0 = MDLSTM_8direction_output(hidden_units[0], input_x, layer=0)
    end1 = time.time()
    
    timespent=(end1-start1)
    print("mdlstmlayer0 time spent: ")
    print(timespent)

    # FC layer 0
    fc_layer_0 = tf.layers.dense(inputs=MDLSTM_layer_0,
                                 units=fc_units[0],
                                 activation=tf.nn.tanh)
    
    # MDLSTM layer 1     
    start2=time.time()
    MDLSTM_layer_1 = MDLSTM_8direction_output(hidden_units[1], fc_layer_0, layer=1)
    end2=time.time()
#     # FC layer 1
#     fc_layer_1 = tf.layers.dense(inputs=MDLSTM_layer_1,
#                                  units=fc_units[1],
#                                  activation=tf.nn.tanh)
    
#     # MDLSTM layer 2    
#     MDLSTM_layer_2 = MDLSTM_8direction_output(hidden_units[2],
#                                               fc_layer_1,
# #                                               index=(2*one_layer_param-2))
#                                               index=2)

    # softmax layer
    fc_layer_2 = tf.layers.dense(inputs=MDLSTM_layer_1,
                                 units=output_size,
                                 activation=tf.nn.softmax)
    
    
    timespent2=(end2-start2)
    print("mdlstmlayer1 time spent: ")
    print(timespent2)

    return fc_layer_2



def cross_entropy(output, input_y):
    input_y=tf.cast(input_y, tf.int64)
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 2)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output))
    return ce

def loss_mse(out, input_y):
    """
    for mse,the second value of softmax can be  used as the probability of a 0-1 problem. So, our mse is calculated based on the value under category "1".
    """
    with tf.name_scope('mean_squared_error'):
        mse = tf.losses.mean_pairwise_squared_error(input_y,out[:,:,:,:,1])
    return mse


# def train_step(mse, learning_rate=1e-3):
#     with tf.name_scope('train_step'):
#         step = tf.train.AdamOptimizer(learning_rate)  #paper use RMSprop, we change it to Adams
#         gvs = step.compute_gradients(mse)      #gradient clipping
#         capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]  
#         training_op = step.apply_gradients(capped_gvs)
#         optimizer= training_op
#     return optimizer


def train_step(error, learning_rate, decay, momentum=0.9, epsilon=1e-5):
    with tf.name_scope('train_step'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate,decay,momentum,epsilon)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(error)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        step = optimizer.apply_gradients(capped_gvs)      
    return step



def evaluate(out):
    with tf.name_scope('evaluate'):
        result=out[:,:,:,:,1]
    return result

# Learned from Assignment 2 cnn.sample
def training(X_train,y_train,
             batch_size,input_size,
             max_step,
             hidden_units=[8, 16],
             fc_units=[25], 
             keep_prob=1.0,
             learning_rate=1e-6+1e-2,
             pre_trained_model=None):
    # define the variables and parameter needed during training
    start = time.time()
    width, height, depth = input_size
    batches = ProcGanerator(batch_size,X_train,y_train,input_size)
    
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [batch_size, width, height, depth, 1])   #channel is 1.
        y = tf.placeholder(tf.int64, [batch_size, width, height, depth])
    out= MDLSTMNet(x,
                                     output_size=2, 
                                     hidden_units=hidden_units, 
                                     fc_units=fc_units,
                                     keep_prob=keep_prob)
    # We can either use mse, or ce for loss.
    loss=loss_mse(out,y)
#     loss=cross_entropy(out,y)
    decay= 1.0
    step=train_step(loss, learning_rate=learning_rate, decay=decay)
    #count the number of parameters
    def count_parameter():
        print ("The number of parameters:")
        print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    cur_model_name = 'mdnet_size{}'.format(width)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/md_{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        count_parameter()
        sess.graph.finalize()
        epoch = 0
        pred_X_val=[]
        pred_y_val=[]
        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, pre_trained_model)
            except Exception:
                raise ValueError("Load model Failed!")
                
        # Train network
        for epoch in range(1,max_step+1):
            x_batch, y_batch = next(batches.batch_generator())
            decay=(1e-2*0.5**(epoch/100)+1e-6)/(1e-2*0.5**((epoch-1)/100)+1e-6)  #use the rule in paper
            _,cur_loss= sess.run([step,loss], feed_dict={x: x_batch, y: y_batch})
#             print('step: {} '.format(epoch),
#                       'loss: {:.4f} '.format(cur_loss))
            if epoch % 1 == 0:
                print('step: {} '.format(epoch),
                      'loss: {:.4f} '.format(cur_loss))
            
            if (epoch % 10 == 0):                   #save model every 500 step is pre-set, we don't change it.
                #save checkpoint and do validation
                saver.save(sess, "checkpoints/md_size{}_step{}.ckpt".format(width,epoch))
                X_val,y_val=batches.get_remain()  #get the validation data from the previous data
                pred_X_val.append(X_val)
                pred_y_val.append(y_val)
                batches = ProcGanerator(batch_size,X_train,y_train,input_size) #renew the batches generator
                
        end = time.time()  
        totaltime=(end-start)
    
        saver.save(sess, 'model/{}'.format(cur_model_name))
        print("Traning ends. Model named {}.".format(cur_model_name))
        print("The total computation time is: {}.".format(totaltime))
        return pred_X_val,pred_y_val

    
def validation(X_val,y_val,
         batch_size,input_size,
         hidden_units=[8,16],
         fc_units=[25],
         keep_prob=1.0,
         learning_rate=1e-3,
         pre_trained_model=''):
    
    # define the variables and parameters needed during training
    val_size=X_val.shape[0]
    width, height, depth = input_size
    x = tf.placeholder(tf.float32, [val_size, width, height, depth, 1])   #channel is 1.
    y = tf.placeholder(tf.int64, [val_size, width, height, depth])
    out= MDLSTMNet(x,y,
                     output_size=2, 
                     hidden_units=hidden_units, 
                     fc_units=fc_units,
                     keep_prob=keep_prob)
    result=evaluate(out)
    with tf.Session() as sess:
        #reload model from either checkpoint or previous models.
        saver=tf.train.Saver()
        saver.restore(sess, pre_trained_model)
        
        # get prediction
        prediction= sess.run([result], feed_dict={x: X_val, y: y_val})
    return prediction


def test(X_test,y_test,
         batch_size,input_size,
         hidden_units=[8, 16],
         fc_units=[25], 
         keep_prob=1.0,
         learning_rate=1e-3,
         pre_trained_model=''):
    
    # define the variables and parameter needed during training
    width, height, depth = input_size
    x = tf.placeholder(tf.float32, [1, width, height, depth, 1])   #channel is 1.
    y = tf.placeholder(tf.int64, [1, width, height, depth])
    out= MDLSTMNet(x,y,
                     output_size=2, 
                     hidden_units=hidden_units, 
                     fc_units=fc_units,
                     keep_prob=keep_prob)
    result_tmp=evaluate_zhx(out)
    result=tf.squeeze(result_tmp)    #squeeze it to a 3D output
    with tf.Session() as sess:
        #reload model from either checkpoint or previous models.
        saver=tf.train.Saver()
        saver.restore(sess, pre_trained_model)
        
        # get prediction
        prediction= sess.run([result], feed_dict={x: X_test, y: y_test})
    return prediction

