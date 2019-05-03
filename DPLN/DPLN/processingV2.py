def Z_ScoreNormalization(x_train,mu,sigma):
    # implement basic Z-score Normalization, get a image which mean=0, var=1
    
    x = (x_train - mu) / sigma;
    
    return x;    
    
    
    
    
    
    
def denoise(raw_image,raw_image1,tolerance=0.1,step=0.125,weight=100):
    import numpy as np
     # This algorithm aims to denoise (minimize the total variation and keep the boundary information)
     # Reference: An Algorithm for Total Variation Minimization and Applications
     # Chambolle, A. Journal of Mathematical Imaging and Vision (2004) 20: 89. https://doi.org/10.1023/B:JMIV.0000011325.36760.1e
     # set loss=||output-previous output||_2 /sqrt(x*y), aiming to minimize it. 
     
     # the input is two raw images, the same with each other. the tolerance, step and weight paramaters are already defined.
     # the output is denoised image
    x_size=raw_image.shape[0]
    y_size=raw_image.shape[1]
    output=raw_image1
    x_component=raw_image 
    y_component=raw_image 
    loss=1  #loss_0=1

    while (loss>tolerance):
        prev_output=output #store the original output 
        gradient_x=np.roll(output,-1,axis=1)-output # gradient, along with 
        gradient_y=np.roll(output,-1,axis=0)-output 
        x_component=x_component+(step/weight)*gradient_x
        y_component=y_component+(step/weight)*gradient_y
        
        power_sum=x_component**2+y_component**2
        update=np.maximum(1,np.sqrt(power_sum))        
        x_component=x_component/update # update
        y_component=y_component/update 

        shift_x=np.roll(x_component,1,axis=1) # shift 
        shift_y=np.roll(y_component,1,axis=0) 

        divergence=(x_component-shift_x)+(y_component-shift_y) # represent the divergence
        output=raw_image+weight*divergence # use weight and divergency to update the output image
          
        loss=np.linalg.norm(output-prev_output,ord=2)/np.sqrt(x_size*y_size); #calculate loss

    return output 
    
def image_augmentation(train_data,label_data,flag):
#     this function is to augment the training data by flipping along the x or y or diagonal axis and by rotating 90 degrees


    import tensorflow as tf
    im=train_data.copy()
    im2=label_data.copy()
    if flag[0]==1:
        with tf.Session() as sess:               
            im_flip_ud=tf.image.flip_up_down(im) #flip the image along to x axis
            im_flip_ud2=tf.image.flip_up_down(im2) 
            im=im_flip_ud.eval()
            im2=im_flip_ud2.eval()
            sess.close()
    if flag[1]==1:
        with tf.Session() as sess:               
            im_flip_lr=tf.image.flip_left_right(im) #flip the image along to y axis
            im=im_flip_lr.eval()
            im_flip_lr2=tf.image.flip_left_right(im2) 
            im2=im_flip_lr2.eval()     
            sess.close()
    if flag[2]==1:
        with tf.Session() as sess:               
            im_transpose=tf.image.transpose_image(im)  #flip the image along to diagonal
            im=im_transpose.eval()
            im_transpose2=tf.image.transpose_image(im2) 
            im2=im_transpose2.eval()
            sess.close()                
    if flag[3]==1:
        with tf.Session() as sess:               
            im_rot=tf.image.rot90(im) #rotate 90 degree
            im=im_rot.eval()
            im_rot2=tf.image.rot90(im2)
            im2=im_rot2.eval()
            sess.close()           
    tf.reset_default_graph()
    return im,im2
    
    
    
def pre_procssing(x_train,y_train):
    import numpy as np
     # this function is to preprocessing the training data and to implement augmentation data.
     # the input is training data X_train and corresponding labels
     # the output has size of [5,512,512,30]; 
    im=np.empty(x_train.shape)
    print('raw image is preprocessing...')
    for i in range(x_train.shape[2]):
        temp=x_train[:,:,i]
        im_d=denoise(temp,temp) #first apply ROF denoise
        mu=np.average(im_d)
        sigma=np.std(im_d)
        im[:,:,i]=Z_ScoreNormalization(im_d,mu,sigma) # then apply z-score normalization
    print('successfully apply ROF denoise and Z-scoring')    
    output_x=np.empty([5,x_train.shape[0],x_train.shape[1],x_train.shape[2]])
    output_y=np.empty([5,x_train.shape[0],x_train.shape[1],x_train.shape[2]])
    output_x[0,:,:,:]=im.copy()    # preprocessed original train data
    output_y[0,:,:,:]=y_train.copy() # original label
    # augment the training data and label simultaneously
    output_x[1,:,:,:],output_y[1,:,:,:]=image_augmentation(im,y_train,[1,0,0,0])
    output_x[2,:,:,:],output_y[2,:,:,:]=image_augmentation(im,y_train,[0,1,0,0])
    output_x[3,:,:,:],output_y[3,:,:,:]=image_augmentation(im,y_train,[0,0,1,0])
    output_x[4,:,:,:],output_y[4,:,:,:]=image_augmentation(im,y_train,[0,0,0,1])
    return output_x,output_y



def test_proc(x_test):
    # process the test data, only implement ROF denoise and Z-scoring
    import numpy as np
    
    im=np.empty(x_test.shape)
    print('test input is preprocessing...')
    for i in range(x_test.shape[2]):
        temp=x_test[:,:,i]
        im_d=denoise(temp,temp)
        mu=np.average(im_d)
        sigma=np.std(im_d)
        im[:,:,i]=Z_ScoreNormalization(im_d,mu,sigma)
    print('test input is preprocessed...')
    return im