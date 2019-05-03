from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from DPLN.pyramidLSTM import PyramidLSTM
from DPLN.ProcGeneratorV4 import ProcGanerator
from tensorflow.python.ops import variable_scope as vs



def PyramidLSTMNet(input_x,
                   output_size=2,
                   hidden_units=[16, 32, 64], fc_units=[25, 45],
                   conv_kernel_size=[7,7], keep_prob=1.0):
    # PyramidLSTMlayer 1
    batch_size= int(input_x.shape[0])
    #test the time for pyramidLSTM
    start = time.time()
    PyLSTM_layer_1 = PyramidLSTM(input_x,1,
                                 hidden_units[0],batch_size,
                                 conv_kernel_size,scope_name="p1")
    end = time.time()
    # FC layer 1
    fc_layer_1 = tf.layers.dense(inputs=PyLSTM_layer_1.output(),
                                 units=fc_units[0],
                                 activation=tf.nn.tanh)   #tanh can change to relu
    
    # PyramidLSTMlayer 2
    PyLSTM_layer_2 = PyramidLSTM(fc_layer_1,fc_units[0],
                                 hidden_units[1],batch_size,
                                 conv_kernel_size,scope_name="p2")
    
    # FC layer 2
    fc_layer_2 = tf.layers.dense(inputs=PyLSTM_layer_2.output(),
                                 units=fc_units[1],
                                 activation=tf.nn.tanh)   #tanh can change to relu 
    
    # PyramidLSTMlayer 3
    PyLSTM_layer_3 = PyramidLSTM(fc_layer_2,fc_units[1],
                                 hidden_units[2],batch_size,
                                 conv_kernel_size,scope_name="p3")
    # Output layer
    out = tf.layers.dense(inputs=PyLSTM_layer_3.output(),
                          units=output_size,
                          activation=tf.nn.softmax)
    timespent=(end-start)
    return out,timespent


def cross_entropy(out, input_y):
    input_y=tf.cast(input_y, tf.int64)
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 2)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=out))
    return ce

def loss_mse(out, input_y):
    """
    for mse,the second value of softmax can be  used as the probability of a 0-1 problem. So, our mse is calculated based on the value under category "1".
    """
    with tf.name_scope('mean_squared_error'):
        mse = tf.losses.mean_squared_error(input_y,out[:,:,:,:,1])
    return mse

# def train_step(error, learning_rate=1e-3):
#     with tf.name_scope('train_step'):
#         step = tf.train.AdamOptimizer(learning_rate).minimize(error)
#     return step
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

#if we use this evaluate function, it may return accuracy, but we are not going to use accuracy 
# def evaluate(out, input_y):
#     input_y=tf.cast(input_y, tf.int64)
#     with tf.name_scope('evaluate'):
#         pred = tf.argmax(output, axis=4, output_type=tf.int64)
#         error_num = tf.count_nonzero(pred - input_y, name='error_num')
#         tf.summary.scalar('MDLSTMNet_error_num', error_num)
#     return error_num

def training(X_train,y_train,
             batch_size,input_size,
             max_step,
             hidden_units=[16, 32, 64],
             fc_units=[25, 45],
             conv_kernel_size=[7,7],
             keep_prob=1.0,
             learning_rate=1e-6+1e-2,
             pre_trained_model=None):
    # define the variables and parameter needed during training
    width, height, depth = input_size
    batches = ProcGanerator(batch_size,X_train,y_train,input_size)
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
    
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [batch_size, width, height, depth, 1])   #channel is 1.
        y = tf.placeholder(tf.int64, [batch_size, width, height, depth])
    with tf.device('/gpu:0'):
        out,timespent= PyramidLSTMNet(x,
                                     output_size=2, 
                                     hidden_units=hidden_units, 
                                     fc_units=fc_units,
                                     conv_kernel_size=conv_kernel_size, 
                                     keep_prob=keep_prob)
    # We can either use mse, or ce for loss.
    with tf.device('/gpu:1'):
        loss=loss_mse(out,y)
#     loss=cross_entropy(out,y)
    
    # Loss calculation is the most memory-requiring part, we assign CPU to calculate independently  
    with tf.device('/cpu'):
        decay= 1.0
        step=train_step(loss, learning_rate=learning_rate, decay=decay)
    #count the number of parameters
    def count_parameter():
        print ("The number of parameters:")
        print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    cur_model_name = 'pyramidnet_size{}'.format(width)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    celltime=[]  #record the time spent in pyramidlayer1
    
    
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":8},inter_op_parallelism_threads=0,
  intra_op_parallelism_threads=0,log_device_placement=True, allow_soft_placement=True,gpu_options=gpu_options))
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
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
        celltime.append(timespent)  #record the time spent in pyramidlayer1 in this step
#             print('step: {} '.format(epoch),
#                       'loss: {:.4f} '.format(cur_loss))
        if epoch % 10 == 0:
            print('step: {} '.format(epoch),
                  'loss: {:.4f} '.format(cur_loss))
            
        if (epoch % 20 == 0):                   #save model every 500 step is pre-set, we don't change it.
            saver.save(sess, "checkpoints/size{}_step{}.ckpt".format(width,epoch))
            X_val,y_val=batches.get_remain()  #get the validation data from the previous data
            pred_X_val.append(X_val)
            pred_y_val.append(y_val)
            batches = ProcGanerator(batch_size,X_train,y_train,input_size) #renew the batches generator
                
            
    saver.save(sess, 'model/{}'.format(cur_model_name))
    print("Traning ends. Model named {}.".format(cur_model_name))
    print("The computation time of the first LSTMlayer is {} seconds.".format(np.mean(celltime)))
    return pred_X_val,pred_y_val

def validation(X_val,y_val,
         batch_size,input_size,
         hidden_units=[16, 32, 64],
         fc_units=[25, 45],
         conv_kernel_size=[7,7],
         keep_prob=1.0,
         learning_rate=1e-6+1e-2,
         pre_trained_model=''):
    
    # define the variables and parameter needed during validation
    val_size=X_val.shape[0]
    width, height, depth = input_size
#     res = np.array(input_size)
    x = tf.placeholder(tf.float32, [val_size, width, height, depth, 1])   #channel is 1.
    y = tf.placeholder(tf.int64, [val_size, width, height, depth])
#     result = tf.placeholder(tf.float32, name="pred_labels") 
    with tf.device('/cpu:0'):
        out,_= PyramidLSTMNet(x,
                             output_size=2, 
                             hidden_units=hidden_units, 
                             fc_units=fc_units,
                             conv_kernel_size=conv_kernel_size, 
                             keep_prob=keep_prob)
    with tf.device('/cpu:7'):
        result=evaluate(out)
        
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    #reload model from either checkpoint or previous models.
    saver=tf.train.Saver()
    saver.restore(sess, pre_trained_model)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)      
    # get prediction
    prediction= sess.run([result], feed_dict={x: X_val, y: y_val},options=run_options)
    return prediction
    
def test(X_test,
         hidden_units=[16, 32, 64],
         fc_units=[25, 45],
         conv_kernel_size=[7,7],
         keep_prob=1.0,
         learning_rate=1e-6+1e-2,
         pre_trained_model=''):
    
    x = tf.placeholder(tf.float32, [1, 512, 512, 30, 1])   #channel is 1.
    with tf.device('/cpu:0'):
        out,_= PyramidLSTMNet(x,
                             output_size=2, 
                             hidden_units=hidden_units, 
                             fc_units=fc_units,
                             conv_kernel_size=conv_kernel_size, 
                             keep_prob=keep_prob)
    with tf.device('/cpu:7'):
        result=tf.squeeze(evaluate(out))    #squeeze it to a 3D output
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":8},inter_op_parallelism_threads=0,
  intra_op_parallelism_threads=0,log_device_placement=True, allow_soft_placement=True,gpu_options=gpu_options))
    #reload model from either checkpoint or previous models.
    saver=tf.train.Saver()
    saver.restore(sess, pre_trained_model)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)    
    # get prediction
    prediction= sess.run([result], feed_dict={x: X_test},options=run_options)
    
    return prediction
        
        
