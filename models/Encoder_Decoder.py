
from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0,stride=1,padding = 'VALID'):

    """
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
    """
    conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None,stride=1)
    #conv = tf.nn.relu(slim.batch_norm(conv, fused=True))
    #conv = slim.conv2d(conv, n_filters, kernel_size, activation_fn=None, normalizer_fn=None,stride=1)
    out = tf.nn.relu(slim.batch_norm(conv, fused=True))
    if dropout_p != 0.0:
        out = slim.dropout(out, keep_prob=(1.0-dropout_p))
    return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0,stride=[2, 2]):
    """
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
    """
    conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    out = tf.nn.relu(slim.batch_norm(conv))
    if dropout_p != 0.0:
        out = slim.dropout(out, keep_prob=(1.0-dropout_p))
    return out
	
def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def VGG19(inputs):
    """
    Impliments of VGG19 Encoder architect
    """   
    #block 1 
    #pre_net = net
    
    
    return net 
'''
def AttentionRefinementModule(inputs,channels, last_arm = False):
  	global_pool = tf.reduce_mean(input, [1, 2], keep_dims=True)
    conv_1 = conv_block(inputs, channels, kernel_size = [1 ,1],stride=1)
    sigmoid = tf.sigmoid(conv_1)
    mul_out = tf.multiply(inputs, sigmoid)
    if last_arm :
      glob_red = tf.reduce_mean(mul_out, [1, 2], keep_dims=True)
      out_scale = tf.multiply(glob_red, mul_out)
      return out_scale
    else :
      return mul_out
    
def FeatureFusionModule(inputs, layer , channels )
	
'''    
  

def TCCN(inputs):
    """
    Impliments of VGG19 Encoder architect
    """   
    
    return net
'''
def build_encoder_decoder(inputs, num_classes, preset_model = "Encoder-Decoder", dropout_p=0.5, scope=None):
	"""
	Builds the Encoder-Decoder model. Inspired by SegNet with some modifications
	Optionally includes skip connections
	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)
	Returns:
	  Encoder-Decoder model
	"""


	if preset_model == "Encoder-Decoder":
		has_skip = False
	elif preset_model == "Encoder-Decoder-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))
  
	#####################
	# Downsampling path #
	#####################
    net = conv_block(inputs, 64, kernel_size = [7 ,7],stride=2)
    net = slim.pool(net, [3, 3], stride=[2, 2], pooling_type='MAX')
    
    
    #block 1
    for i in range(2):
      if i == 0:
        temp =  conv_block(net, 64*4, kernel_size = [1 ,1])
      else:
        temp = net
      net = conv_block(net, 64, kernel_size = [1 ,1])
      net = conv_block(net, 64, kernel_size = [3 ,3])
      net = conv_block(net, 64*4,kernel_size = [1 ,1])
      net = tf.add(net,temp)
      net = tf.nn.relu(net)  
	layer_1 = net
	
    
    #block 2 
	for i in range(3):
      if i == 0:
        temp =  conv_block(net, 128*4, kernel_size = [1 ,1])
      else:
        temp = net
      net = conv_block(net, 128, kernel_size = [1 ,1])
      net = conv_block(net, 128, kernel_size = [3 ,3])
      net = conv_block(net, 128*4,kernel_size = [1 ,1])
      net = tf.add(net,temp)
      net = tf.nn.relu(net)  
	layer_2 = net
    
    #block 3
	for i in range(22):
      if i == 0:
        temp =  conv_block(net, 256*4, kernel_size = [1 ,1])
      else:
        temp = net
      net = conv_block(net, 256, kernel_size = [1 ,1])
      net = conv_block(net, 256, kernel_size = [3 ,3],padding = 'SAME')
      net = conv_block(net, 256*4,kernel_size = [1 ,1])
      net = tf.add(net,temp)
      net = tf.nn.relu(net)  
	layer_3 = net
    
    
    #block 4
	for i in range(2):
      if i == 0:
        temp =  conv_block(net, 512*4, kernel_size = [1 ,1])
      else:
        temp = net
      net = conv_block(net, 512, kernel_size = [1 ,1])
      net = conv_block(net, 512, kernel_size = [3 ,3])
      net = conv_block(net, 512*4,kernel_size = [1 ,1])
      net = tf.add(net,temp)
      net = tf.nn.relu(net)  
	layer_4 = net
    
    
    
	


	#####################
	# Upsampling path #
	#####################
	
    
    
    
    
	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net

def build_encoder_decoder(inputs, num_classes, preset_model = "Encoder-Decoder", dropout_p=0.5, scope=None):
	"""
	Builds the Encoder-Decoder model. Inspired by SegNet with some modifications
	Optionally includes skip connections
	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)
	Returns:
	  Encoder-Decoder model
	"""


	if preset_model == "Encoder-Decoder":
		has_skip = False
	elif preset_model == "Encoder-Decoder-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))

	#####################
	# Downsampling path #
	#####################
	net_plus = conv_block(inputs, 64, kernel_size = [3 ,3])
	net = conv_block(inputs, 64)
	net = conv_block(net, 64)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1 = net
	
	
	net_plus = conv_block(net,128,kernel_size=[3,3])
	net = conv_block(net, 128)
	net = conv_block(net, 128)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2 = net

	net_plus = conv_block(net, 256, kernel_size = [3 ,3])
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_3 = net

	net_plus = conv_block(net, 512, kernel_size = [3 ,3])
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_4 = net
	net_plus = conv_block(net, 1024, kernel_size = [3 ,3])
	net = conv_block(net, 1024)
	net = conv_block(net, 1024)
	net = conv_block(net, 1024)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	#####################
	# Upsampling path #
	#####################
	net = conv_transpose_block(net, 512)
	net_plus = conv_block(net, 512, kernel_size = [3 ,3])
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	if has_skip:
		net = tf.add(net, skip_4)

	net = conv_transpose_block(net, 512)
	net_plus = conv_block(net, 256, kernel_size = [3 ,3])
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 256)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	if has_skip:
		net = tf.add(net, skip_3)

	net = conv_transpose_block(net, 256)
	net_plus = conv_block(net, 128, kernel_size = [3 ,3])
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 128)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	if has_skip:
		net = tf.add(net, skip_2)

	net = conv_transpose_block(net, 128)
	net_plus = conv_block(net, 64, kernel_size = [3 ,3])
	net = conv_block(net, 128)
	net = conv_block(net, 64)
	net = tf.add(net,net_plus)
	net = tf.nn.relu(net)
	if has_skip:
		net = tf.add(net, skip_1)

	net = conv_transpose_block(net, 64)
	net = conv_block(net, 64)
	net = conv_block(net, 64)

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net
'''  
def build_encoder_decoder(inputs, num_classes, preset_model = "Encoder-Decoder", dropout_p=0.5, scope=None):
    if preset_model == "Encoder-Decoder":
        has_skip = False
    elif preset_model == "Encoder-Decoder-Skip":
        has_skip = True
    else:
        raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))

	#####################
	# Encoder path #
	#####################
    net = conv_block(inputs,64)
    pre_net = net
    net = conv_block(net,64)
    net = conv_block(net,64)
    net = tf.add(net,pre_net) 
    skip1 = net
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    #block 2
    net = conv_block(net,128)
    pre_net = net
    net = conv_block(net,128)
    net = conv_block(net,128)
    net = tf.add(net,pre_net)
    skip2 = net
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    #block 3 
    net = conv_block(net,256)
    pre_net = net
    net = conv_block(net,256)
    net = conv_block(net,256)
    net = tf.add(net,pre_net)
    skip3 = net
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    #block 4
    net = conv_block(net,512)
    pre_net = net
    net = conv_block(net,512)
    net = conv_block(net,512)
    net = tf.add(net,pre_net)
    skip4 = net
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    
    net = conv_block(net,1024,dropout_p=0.5)
    pre_net = net
    net = conv_block(net,1024,dropout_p=0.5)
    net = conv_block(net,1024,dropout_p=0.5)
    net = tf.add(net,pre_net)
    
    net = conv_transpose_block(net,512)
    net = tf.add(net,skip4)

    net = conv_block(net,512)
    pre_net = net
    net = conv_block(net,512)
    net = conv_block(net,512)
    net = tf.add(net,pre_net)

    net = conv_transpose_block(net,256)
    net = tf.add(net,skip3)

    net = conv_block(net,256)
    pre_net = net
    net = conv_block(net,256)
    net = conv_block(net,256)
    net = tf.add(net,pre_net)

    net = conv_transpose_block(net,128)
    net = tf.add(net,skip2)

    net = conv_block(net,128)
    pre_net = net
    net = conv_block(net,128)
    net = conv_block(net,128)
    net = tf.add(net,pre_net)
    
    net = conv_transpose_block(net,64)
    net = tf.add(net,skip1)
    
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net       
