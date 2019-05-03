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
from tensorflow.python.ops import variable_scope as vs

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
    
class MDLSTMcell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=tf.AUTO_REUSE,
                 name=None,
                 dtype=dtypes.float32):
        """
        Adapted from TensorFlow's BasicLSTMCell. state_is_tuple is always True.
        
        :param num_units: hidden units number
        :param forget_bias: used for adding in "call" function
        :param state_is_tuple: if True, the state=(c,h)
        :param activation: The activation function in "call" function
        :param reuse: reuse kernel and bias variable
        :param name: the name of current cell
        """

        super(MDLSTMcell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple=state_is_tuple
        self._activation = activation or math_ops.tanh
    
    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2*self._num_units)

    @property
    def output_size(self):
        return self._num_units
    
    def call(self, inputs, state):
        """
        Multi-Dimensional Long short-term memory cell (MDLSTMcell), revised based on TensorFlow BasicLSTMCell.
        
        :param inputs: 5D tensor, have size of (batch_size, img_len, img_len, img_dep ,channels)
        :param state: the tuple state (c,h) of the previous cell
        
        :return
        new_h: h of current cell.
        new_state: Tuple state of current cell.
        """
        
        inputs_shape=inputs.shape
    
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))

        input_depth = inputs_shape[1].value
        #this cell needs h comes from 3 dimensions
        h1_depth = self._num_units
        h2_depth = self._num_units
        h3_depth = self._num_units
        # create variables for kernel and bias
        with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            self._kernel = self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth+h1_depth+h2_depth+h3_depth, 6*self._num_units])
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[6*self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))
        
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        
        # get the c and h from 3 dimensions respectively
        state1=state[0]
        state2=state[1]
        state3=state[2]
        c1, h1=state1
        c2, h2=state2
        c3, h3=state3
        
        # Parameters of gates are concatenated into one multiply for efficiency.
        gate_inputs=math_ops.matmul(array_ops.concat([inputs, h1, h2, h3], 1), self._kernel)
        gate_inputs=nn_ops.bias_add(gate_inputs, self._bias)
        
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f1, f2, f3, o=array_ops.split(value=gate_inputs, num_or_size_splits=6, axis=one)
        forget_bias_tensor=constant_op.constant(self._forget_bias, dtype=f1.dtype)
        
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c=add(multiply(c1, sigmoid(add(f1, forget_bias_tensor))),
                  multiply(c2, sigmoid(add(f2, forget_bias_tensor))))
        new_c=add(new_c, multiply(c3, sigmoid(add(f3, forget_bias_tensor))))
        new_c=add(new_c, multiply(sigmoid(i), self._activation(j)))
                  
#         new_c=(c1 * sigmoid(f1 + forget_bias_tensor) + 
#                c2 * sigmoid(f2 + forget_bias_tensor) + 
#                c3 * sigmoid(f3 + forget_bias_tensor) +
#                sigmoid(i) * self._activation(j))
        
        new_h=multiply(self._activation(new_c), sigmoid(o))
        
        if self._state_is_tuple:
            new_state=tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state=array_ops.concat([new_c, new_h], 1)
        
        return new_h, new_state

    
def MDLSTMcell_whileloop(rnn_size,input_data,index):
    

    # j used for naming only
    j=index
    batch_size, img_len, img_len, z_dim, channels=input_data.shape
    flag_cun=0  # flag number
    # i used for locating cell
    i=-2
    
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        for x3 in range(z_dim):
            for x1 in range(img_len):
                for x2 in range(img_len):
                    ####################################################
                    i=i+1
                    cell=MDLSTMcell(rnn_size,name='md_lstm_cell_{}_{}'.format(j,i),reuse=tf.AUTO_REUSE)
                    ####################################################
                    if x3==0:
                        if x1==0 and x2==0:
                            state1=cell.zero_state(batch_size, tf.float32)
                            state2=cell.zero_state(batch_size, tf.float32)
                            state3=cell.zero_state(batch_size, tf.float32)

                        elif x2!=0 and x1==0:
                            state1=cell.zero_state(batch_size, tf.float32)
                            state2=tf.contrib.rnn.LSTMStateTuple(all_c[:,i,:], all_h[:,i,:]) #传入x2方向上一个
                            state3=cell.zero_state(batch_size, tf.float32)

                        elif x1!=0 and x2==0:
                            state1=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len-1),:], all_h[:,i-(img_len-1),:]) #传入x1方向上一个
                            state2=cell.zero_state(batch_size, tf.float32)
                            state3=cell.zero_state(batch_size, tf.float32)

                        elif x1!=0 and x2!=0:
                            state1=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len-1),:], all_h[:,i-(img_len-1),:]) #传入x1方向上一个
                            state2=tf.contrib.rnn.LSTMStateTuple(all_c[:,i,:], all_h[:,i,:]) #传入x2方向上一个
                            state3=cell.zero_state(batch_size, tf.float32)

                    if x3!=0:
                        if x1==0 and x2==0:
                            state1=cell.zero_state(batch_size, tf.float32)
                            state2=cell.zero_state(batch_size, tf.float32)
                            state3=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len*img_len-1),:], all_h[:,i-(img_len*img_len-1),:]) #传入x3方向上一个

                        elif x2!=0 and x1==0:
                            state1=cell.zero_state(batch_size, tf.float32)
                            state2=tf.contrib.rnn.LSTMStateTuple(all_c[:,i,:], all_h[:,i,:])
                            state3=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len*img_len-1),:], all_h[:,i-(img_len*img_len-1),:])

                        elif x1!=0 and x2==0:
                            state1=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len-1),:], all_h[:,i-(img_len-1),:])
                            state1=cell.zero_state(batch_size, tf.float32)
                            state3=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len*img_len-1),:], all_h[:,i-(img_len*img_len-1),:]) #传入x3方向上一个                

                        elif x1!=0 and x2!=0:
                            state1=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len-1),:], all_h[:,i-(img_len-1),:])
                            state2=tf.contrib.rnn.LSTMStateTuple(all_c[:,i,:], all_h[:,i,:])
                            state3=tf.contrib.rnn.LSTMStateTuple(all_c[:,i-(img_len*img_len-1),:], all_h[:,i-(img_len*img_len-1),:]) #传入x3方向上一个       

                    # get previous cell state for current cell to use as initial_state
                    state=(state1, state2, state3)

                    # define input
                    input_t=input_data[:,x1,x2,x3,:]

                    # implement dynamic_rnn to get (c,h)  Too much time cost and therefore deprecated
#                     outputs, new_state=tf.nn.dynamic_rnn(cell, input_t, initial_state=state,
#                                                          dtype=dtypes.float32,scope='md_lstm_state_{}_{}'.format(j,x3))
                    outputs, new_state=cell.call(input_t, state)

                    # save the output of all cells
                    if flag_cun!=0:
                        outputs=tf.reshape(outputs, [batch_size,1,rnn_size])
                        all_h=tf.concat([all_h,outputs], axis=1)
                        new_state=tf.reshape(new_state[0], [batch_size,1,rnn_size])
                        all_c=tf.concat([all_c,new_state], axis=1)

                    if flag_cun==0:
                        all_h=tf.reshape(outputs, [batch_size,1,rnn_size])
                        all_c=tf.reshape(new_state[0], [batch_size,1,rnn_size])
                        flag_cun=1

    # To make the output of all cell in order
    cell_out = tf.reshape(all_h, (batch_size, z_dim, img_len, img_len, rnn_size))
    cell_out = tf.transpose(cell_out, (0,2,3,1,4))

    return cell_out

def MDLSTM_8direction_output(rnn_size,input_x,layer):

    # Input_x shape: (bs, x,y,z, c)
    # These transformations all deal with (bs, z,x,y, c) so we need to transform it again.

    x1_out=input_x # 1->48  No need transform again
    out1=MDLSTMcell_whileloop(rnn_size, x1_out, index=layer) # (bs, x,y,z, c)

    x1=input_x     
    x1=tf.transpose(x1, (0,3,1,2,4)) # prepare for transformations below (bs, z,x,y, c) 

    x2_old=tf.reverse(x1,axis=[2])  
    x2=tf.transpose(x2_old, (0,1,3,2,4))  # 13->36
    x2_out=tf.transpose(x2, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out2=MDLSTMcell_whileloop(rnn_size, x2_out, index=layer) # (bs, x,y,z, c)
    out2=tf.transpose(out2, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out2=tf.transpose(out2, (0,1,3,2,4))
    out2=tf.reverse(out2,axis=[2])  # now change to the direction 1->48 (bs, z,x,y, c)
    out2=tf.transpose(out2, (0,2,3,1,4)) # (bs, x,y,z, c)


    x3=tf.reverse(x2_old,axis=[3])  # 16->33
    x3_out=tf.transpose(x3, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out3=MDLSTMcell_whileloop(rnn_size, x3_out, index=layer) # (bs, x,y,z, c)
    out3=tf.transpose(out3, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out3=tf.reverse(out3,axis=[3])
    out3=tf.reverse(out3,axis=[2]) # now change to the direction 1->48 (bs, z,x,y, c)
    out3=tf.transpose(out3, (0,2,3,1,4)) # (bs, x,y,z, c)

    x4=tf.transpose(x1, (0,1,3,2,4))
    x4=tf.reverse(x4, axis=[2])  # 4->45
    x4_out=tf.transpose(x4, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out4=MDLSTMcell_whileloop(rnn_size, x4_out, index=layer) # (bs, x,y,z, c)
    out4=tf.transpose(out4, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out4=tf.reverse(out4, axis=[2])
    out4=tf.transpose(out4, (0,1,3,2,4)) # now change to the direction 1->48 (bs, z,x,y, c)
    out4=tf.transpose(out4, (0,2,3,1,4)) # (bs, x,y,z, c)

    #####################################
    
    x5=tf.reverse(x1,axis=[1])  # 33->16
    x5_out=tf.transpose(x5, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out5=MDLSTMcell_whileloop(rnn_size, x5_out, index=layer) # (bs, x,y,z, c)
    out5=tf.transpose(out5, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out5=tf.reverse(out5,axis=[1]) # now change to the direction 1->48 (bs, z,x,y, c)
    out5=tf.transpose(out5, (0,2,3,1,4)) # (bs, x,y,z, c)


    x6=tf.reverse(x2,axis=[1])  # 45->4
    x6_out=tf.transpose(x6, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out6=MDLSTMcell_whileloop(rnn_size, x6_out, index=layer) # (bs, x,y,z, c)
    out6=tf.transpose(out6, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out6=tf.reverse(out6,axis=[1])  
    out6=tf.transpose(out6, (0,1,3,2,4)) 
    out6=tf.reverse(out6,axis=[2])  # now change to the direction 1->48 (bs, z,x,y, c)
    out6=tf.transpose(out6, (0,2,3,1,4)) # (bs, x,y,z, c)


    x7=tf.reverse(x3,axis=[1])  # 48->1
    x7_out=tf.transpose(x7, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out7=MDLSTMcell_whileloop(rnn_size, x7_out, index=layer) # (bs, x,y,z, c)
    out7=tf.transpose(out7, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out7=tf.reverse(out7,axis=[1])  
    out7=tf.reverse(out7,axis=[3])  
    out7=tf.reverse(out7,axis=[2])  # now change to the direction 1->48 (bs, z,x,y, c)
    out7=tf.transpose(out7, (0,2,3,1,4)) # (bs, x,y,z, c)


    x8=tf.reverse(x4,axis=[1])  # 36->13
    x8_out=tf.transpose(x8, (0,2,3,1,4)) # transform again to meet the shape (bs, x,y,z, c)
    out8=MDLSTMcell_whileloop(rnn_size, x8_out, index=layer) # (bs, x,y,z, c)
    out8=tf.transpose(out8, (0,3,1,2,4)) # transform to (bs, z,x,y, c)
    ###
    out8=tf.reverse(out8,axis=[1])  
    out8=tf.reverse(out8,axis=[2])  
    out8=tf.transpose(out8, (0,1,3,2,4)) # now change to the direction 1->48 (bs, z,x,y, c)
    out8=tf.transpose(out8, (0,2,3,1,4)) # (bs, x,y,z, c)

    # sum all 8 direction outputs
    out_all=out1+out2+out3+out4+out5+out6+out7+out8
#     print(out_all.shape)
    
    return out_all


