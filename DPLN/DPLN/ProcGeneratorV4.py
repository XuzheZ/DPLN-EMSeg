from __future__ import print_function
import random
    
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
import pickle

class ProcGanerator(object):

    # this class is refered as a batch generator. 
    
    
    def __init__(self,batch_size,x_train,y_train,input_size):
        
        # put the date into self to reuse in the following functions
        # batch_size refers to the number of batches, if it is = 1, generating 1 batch 
        # x_train & y_train refer to training data and corresponding labels
        # input_size refers to the size of batch, it should be a [x,y,z]      
        self.x_train=x_train
        self.y_train=y_train
        self.input_size=input_size
        self.batch_size=batch_size
        x,y,z=input_size[0],input_size[1],input_size[2]
        
        
        if len(x_train.shape)==4: #chech the dimension of training data
            X,Y,Z=x_train.shape[1],x_train.shape[2],x_train.shape[3]
        elif len(x_train.shape)==3:
            X,Y,Z=x_train.shape
        else:
            raise ValueError('please check the dimension of x_train')
        if x>X or y>Y or z>Z:
            raise ValueError('batch size exceeds the data size')
        # generate the random coordinate for validation set, save it into self
        self.remain_x=random.randint(0,X-x-1)
        self.remain_y=random.randint(0,Y-y-1)
        self.remain_z=random.randint(0,Z-z-1)
        
    
    def get_remain(self):
        
            
        batch_size=self.batch_size
        x_train=self.x_train
        y_train=self.y_train
        input_size=self.input_size        
        remain_x=self.remain_x
        remain_y=self.remain_y
        remain_z=self.remain_z
        x,y,z=input_size[0],input_size[1],input_size[2]
        if len(x_train.shape)==4:
            
            remain_voxel=x_train[0,remain_x:remain_x+x,remain_y:remain_y+y, remain_z:remain_z+z].copy()
            remain_label=y_train[0,remain_x:remain_x+x,remain_y:remain_y+y, remain_z:remain_z+z].copy()
        elif len(x_train.shape)==3:
            
            remain_voxel=x_train[remain_x:remain_x+x,remain_y:remain_y+y, remain_z:remain_z+z].copy()
            remain_label=y_train[remain_x:remain_x+x,remain_y:remain_y+y, remain_z:remain_z+z].copy()
        print('generated a validation set, the size is [x,y,z], it is : X_train[%d:%d,%d:%d,%d:%d]'%(remain_x,remain_x+x,remain_y,remain_y+y,remain_z,remain_z+z))
        return remain_voxel,remain_label    
    
    
    def batch_generator(self):
        # get the parameters from self
        batch_size=self.batch_size
        x_train=self.x_train
        y_train=self.y_train
        input_size=self.input_size
        remain_x=self.remain_x
        remain_y=self.remain_y
        remain_z=self.remain_z                  
        x,y,z=input_size[0],input_size[1],input_size[2]
        
        
        batch=np.empty([batch_size,x,y,z])
        batch_label=np.empty([batch_size,x,y,z])
        if y_train.all()==None:
            raise ValueError('no label data')
        if len(x_train.shape)==4:
            X,Y,Z=x_train.shape[1],x_train.shape[2],x_train.shape[3]
        elif len(x_train.shape)==3:
            X,Y,Z=x_train.shape
        else:
            raise ValueError('please check the dimension of x_train')
        if x>X or y>Y or z>Z:
            raise ValueError('batch size exceeds the data size')
              
        while True:
            i=0
            
            while i<batch_size:
                
                x_seed=random.randint(0,X-x-1)
                y_seed=random.randint(0,Y-y-1)
                z_seed=random.randint(0,Z-z-1)
                flag=random.randint(0,4)
                if x_seed==remain_x and y_seed==remain_y and z_seed==remain_z and flag==0:
                    i-=1
                else:
                    if len(x_train.shape)==4:
                        temp_x=x_train[flag,x_seed:x_seed+x,y_seed:y_seed+y, z_seed:z_seed+z].copy()
                        temp_y=y_train[flag,x_seed:x_seed+x,y_seed:y_seed+y, z_seed:z_seed+z].copy()
                    elif len(x_train.shape)==3:
                        temp_x=x_train[x_seed:x_seed+x,y_seed:y_seed+y, z_seed:z_seed+z].copy()
                        temp_y=y_train[x_seed:x_seed+x,y_seed:y_seed+y, z_seed:z_seed+z].copy()
                    batch[i,:,:,:]=temp_x.copy()
                    batch_label[i,:,:,:]=temp_y.copy()
                    i+=1
        
            batch=batch.reshape([batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1])
            yield batch,batch_label



