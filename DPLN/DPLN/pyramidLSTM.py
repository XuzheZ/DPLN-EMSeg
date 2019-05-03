from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from cell.rnn_cell import ConvLSTMCell
from tensorflow.python.ops import variable_scope as vs
#记得删掉！！！！！！！！！！！！！！！！！！！！！！！！！！！
# def verbose(original_function):
#     # make a new function that prints a message when original_function starts and finishes
#     def new_function(*args, **kwargs):
#         print('get variable:', '/'.join((tf.get_variable_scope().name, args[0])))
#         result = original_function(*args, **kwargs)
#         return result
#     return new_function

# vs.get_variable = verbose(vs.get_variable)

class PyramidLSTM(object):
    def __init__(self, input_x, in_channel, out_channel, batch_size, kernel_shape, scope_name="p"):
        """
        :param input_x: The input of the conv layer. Should be a 5D array like (batch_num, time_step, img_width, img_height ,channel_num)
        :param in_channel: The 5-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 5-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: The shape of the kernel. Default is [7,7], it means you have a 7*7 kernel.
        """
        self.input_x=input_x                    #(b,w,h,d,c)
        self.in_channel = in_channel            #c
        self.out_channel= out_channel
        self.batch_size = batch_size            #(b)
        self.width=input_x.shape[1]             #w, set to be an even number
        self.height=input_x.shape[2]            #h, set to be an even number
        self.depth = input_x.shape[3]           #d, can be odd or even
        self.kernel_shape= kernel_shape 
        self.scope_name = scope_name
#         self.if_initial = True
        
        self.padding_image()                 #pad the images to ensure that pyramid ends up with 1 pixel's, then to pad the depth to make 3 dimensions the same for convenience
        self.pyramidLayer()                  #lstm layers
        self.cut_image()                     #Cut the padded part to ensure the output has the same image size as the input
        
    def padding_image(self):
        """
        Pad the images to ensure that pyramid ends up with 1 pixel's, then to pad the depth to make 3 dimensions the same for convenience.
        """
        input_x=self.input_x                 #(b,w,h,d,c)
        padded_image=input_x
        if (self.depth % 2==0):
            padding = tf.constant([[0,0,],[1,2,],[1,2,],[1,2,],[0,0]])
        else:
            padding = tf.constant([[0,0,],[1,2,],[1,2,],[1,1,],[0,0]])
            
        padded_image = tf.pad(padded_image,padding,"CONSTANT")
        self.padded_image=padded_image
        self.padded_image_shape = padded_image.get_shape().as_list()[1:4]
        
      
    def pyramidLayer(self):
        
        def split_image(direction,is_reverse=False):
            """
            :direction: The direction chosen as time_step, can be 'w', 'h', 'd'.
            :is_reverse: Bool. If it's True, reverse the image to realize the reverse direction.
            """ 
            padded_image = self.padded_image
            if direction=='w':
                time_steps = self.padded_image_shape[0]
                splited_image = tf.transpose(padded_image,[0,1,2,3,4])
            elif direction=='h':
                time_steps = self.padded_image_shape[1]
                splited_image = tf.transpose(padded_image,[0,2,3,1,4])
            elif direction=='d':
                time_steps = self.padded_image_shape[2]
                splited_image = tf.transpose(padded_image,[0,3,1,2,4])
            if is_reverse:
                splited_image = tf.reverse(splited_image[:,:,:,:,:],axis=[1])
            return splited_image


        def pyramidCell(input_part,index):
            """
            :input_part: The input of the pyramidCell. Should be a volume representing one of all the 6 directions. The shape should be (b,Z,X,Y,c). Comes from function split_image.
            :param index: The index of the pyramidCell, it can be 1-6.
            """
            kernel_shape=self.kernel_shape
            out_channel= self.out_channel
            input_part_shape=input_part.get_shape().as_list()         #[b,Z,X,Y,c]
            time_steps=input_part_shape[1]
            scope_name='{}_{}'.format(self.scope_name,index)
            cells = []
            with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
                cell=ConvLSTMCell(conv_ndims=2,
                                  input_shape=input_part_shape[2:],
                                  output_channels=out_channel,
                                  kernel_shape=kernel_shape,
                                  use_bias=True,
                                  skip_connection=False,
                                  forget_bias=1.0,
                                  initializers=None,
                                  reuse=tf.get_variable_scope().reuse)

                output,_=tf.nn.dynamic_rnn(cell=cell,inputs=input_part,dtype=tf.float32,scope=scope_name)
            return output
        
        with tf.variable_scope(self.scope_name,reuse=False):
            with tf.device('/gpu:0'):
                input_part1=split_image(direction='d',is_reverse=False)  #(z=d,x=w,y=h)
                clstm1=pyramidCell(input_part1,1)
                clstm1 = tf.transpose(clstm1,[0,2,3,1,4])                 #(w,h,d)
            with tf.device('/gpu:1'):
                input_part2=split_image(direction='d',is_reverse=True)   #(z=-d,x=w,y=h)
                clstm2=pyramidCell(input_part2,2)
                clstm2 = tf.reverse(clstm2[:,:,:,:,:],axis=[1])           #(d,w,h)
                clstm2 = tf.transpose(clstm2,[0,2,3,1,4])                 #(w,h,d)
            with tf.device('/gpu:2'):
                input_part3=split_image(direction='w',is_reverse=False)  #(z=w,x=h,y=d)
                clstm3=pyramidCell(input_part3,3)                        #(w,h,d)
            with tf.device('/gpu:3'):
                input_part4=split_image(direction='w',is_reverse=True)   #(z=-w,x=h,y=d)
                clstm4=pyramidCell(input_part4,4)
                clstm4 = tf.reverse(clstm4[:,:,:,:,:],axis=[1])           #(w,h,d)
            with tf.device('/gpu:4'):
                input_part5=split_image(direction='h',is_reverse=False)  #(z=h,x=d,y=w)
                clstm5=pyramidCell(input_part5,5)
                clstm5 = tf.transpose(clstm5,[0,3,1,2,4])                 #(w,h,d)
            with tf.device('/gpu:5'):
                input_part6=split_image(direction='h',is_reverse=True)   #(z=-h,x=d,y=w)
                clstm6=pyramidCell(input_part6,6)
                clstm6 = tf.reverse(clstm6[:,:,:,:,:],axis=[1])           #(h,d,w)
                clstm6 = tf.transpose(clstm6,[0,3,1,2,4])                 #(w,h,d)
        with tf.device('/cpu:6'):
            output=clstm1+clstm2+clstm3+clstm4+clstm5+clstm6
            self.cell_out_orig=output
        return output

    def cut_image(self):
        """
        Cut the padded part to ensure the output has the same image size as the input.
        """
        output=self.cell_out_orig                #(b,w,h,d,c)
        if (self.depth % 2==0):
            cutted_image= output[:,1:-2, 1:-2, 1:-2, :]
        else:
            cutted_image= output[:,1:-2, 1:-2, 1:-1, :]
            
        self.cell_out=cutted_image
        return cutted_image
    
    def output(self):
        return self.cell_out


