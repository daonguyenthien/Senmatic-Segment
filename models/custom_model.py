from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from builders import frontend_builder

def AtrousSpatialPyramidPoolingModule(inputs, depth=256):
    """

    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper

    """

    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
    image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

    atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

    atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)

    atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)

    atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

    atrous_pool_block_24 = slim.conv2d(inputs, depth, [3, 3], rate=24, activation_fn=None)

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18,atrous_pool_block_24), axis=3)
    net = tf.image.resize(net,(256,256))
    net = tf.nn.relu(net)
    net = slim.conv2d(net, depth, [1, 1], scope="logits", activation_fn=None)
    

    return net

def conv_block(inputs, n_filters, filter_size=[3, 3], dropout_p=0.0,strides = 1 ):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    Dropout (if dropout_p > 0) on the inputs
    """
    conv = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None, normalizer_fn=None,stride = strides)
    out = tf.nn.relu(slim.batch_norm(conv, fused=True))
    if dropout_p != 0.0:
      out = slim.dropout(out, keep_prob=(1.0-dropout_p))
    return out

def conv_transpose_block(inputs, n_filters, strides=2, filter_size=[3, 3], dropout_p=0.0):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    Dropout (if dropout_p > 0) on the inputs
    """
    conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[strides, strides])
    out = tf.nn.relu(slim.batch_norm(conv, fused=True))
    if dropout_p != 0.0:
      out = slim.dropout(out, keep_prob=(1.0-dropout_p))
    return out
  
def ARM_block(inputs,channels, last_arm = False):
    """
    Basic ARM Func
    """
    global_pool = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    conv_1 = slim.conv2d(inputs, channels, kernel_size= [1,1], activation_fn=None, normalizer_fn=None,stride=1)
    sigmoid = tf.sigmoid(conv_1)
    mul_out = tf.multiply(inputs, sigmoid)
    if last_arm :
      glob_red = tf.reduce_mean(mul_out, [1, 2], keep_dims=True)
      out_scale = tf.multiply(glob_red, mul_out)
      return out_scale
    else :
      return mul_out
    
def feature_fusion(inputs, skip):
    """
    feature_fusion
    """
    upsample_skip = conv_transpose_block(skip, strides = 2 , n_filters=256)
    
    concat_1 = tf.concat([upsample_skip,inputs],axis = -1)
    conv_1 = conv_block(concat_1,1024,filter_size= [3,3],strides = 1)
    global_pool = tf.reduce_mean(conv_1, [1, 2], keep_dims=True)
    conv_2 = slim.conv2d(global_pool ,1024,kernel_size = [3,3],stride = 1)
    conv_3 = slim.conv2d(conv_2,1024,kernel_size = [3,3],stride = 1)
    sigmoid = tf.sigmoid(conv_3)
    mul = tf.multiply(conv_1, sigmoid)
    add_out = tf.add(conv_1, mul)
    return add_out

def build_custom(inputs, num_classes, frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
	

    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, is_training=is_training)
    arm1 = ARM_block(end_points["pool5"],2048)
    arm1 = tf.keras.layers.UpSampling2D()(arm1)
    
    arm2 = ARM_block(end_points["pool4"],512,last_arm = True)
    arm2 = slim.conv2d(arm2 ,2048,kernel_size = [3,3],stride = 1)
    
    add_arm = tf.add(arm1,arm2)
    ffm1 = feature_fusion(end_points["pool3"],add_arm)
    ASPP=AtrousSpatialPyramidPoolingModule(ffm1,depth = num_classes)
    


    
    return ASPP