from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import nn

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
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
#with RES block
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
	#################################################
	# Downsampling path & Upsampling path &Softmax  #
	#################################################
	#768*384*3
	res_0_1 = conv_block(inputs,64,kernel_size = [3,3])
	res_0_2 = conv_block(res_0_1,64,kernel_size=[3,3])

	pool_0_1 = conv_block(inputs, 64, kernel_size=[3, 3])
	pool_0_2 = conv_block(pool_0_1, 64, kernel_size=[3, 3])

	pool_1_1 = slim.pool(pool_0_2,[2,2],stride = [2,2],pooling_type = 'MAX')
	pool_1_2 = conv_block(pool_1_1,128,kernel_size = [3,3])
	pool_1_3 = conv_block(pool_1_2, 128, kernel_size=[3, 3])

	pool_2_1 = slim.pool(pool_1_3, [2, 2], stride=[2, 2], pooling_type='MAX')
	pool_2_2 = conv_block(pool_2_1, 256 , kernel_size=[3, 3])
	pool_2_3 = conv_block(pool_2_2, 256, kernel_size=[3, 3])

	pool_3_1 = slim.pool(pool_2_3, [2, 2], stride=[2, 2], pooling_type='MAX')
	pool_3_2 = conv_block(pool_3_1, 512, kernel_size=[3, 3])
	pool_3_3 = conv_block(pool_3_2, 512, kernel_size=[3, 3])

	pool_4_1 = slim.pool(pool_3_3, [2, 2], stride=[2, 2], pooling_type='MAX')
	pool_4_2 = conv_block(pool_4_1, 1024, kernel_size=[3, 3])
	pool_4_3 = conv_block(pool_4_2, 1024, kernel_size=[3, 3])
	##
	up_4 = conv_transpose_block(pool_4_3,512)
	plus_4 = tf.add(up_4,pool_4_1)
	res_4_1 = conv_block(plus_4, 512, kernel_size=[3, 3])
	res_4_2 = conv_block(res_4_1, 512 , kernel_size=[3, 3])
	res_4_3 = conv_block(res_4_2, 512, kernel_size=[3, 3])
	res_4_1_plus = tf.add (res_4_1,res_4_3)
	res_4_4 = conv_block(res_4_1_plus, 512, kernel_size=[3, 3])
	res_4_5 = conv_block(res_4_4, 512, kernel_size=[3, 3])
	res_4_2_plus = tf.add (res_4_1_plus, res_4_5)

	up_3 = conv_transpose_block(res_4_2_plus, 256)
	plus_3 = tf.add(up_3,pool_3_1)
	res_3_1 = conv_block(plus_3, 256, kernel_size=[3, 3])
	res_3_2 = conv_block(res_3_1, 256, kernel_size=[3, 3])
	res_3_3 = conv_block(res_3_2, 256, kernel_size=[3, 3])
	res_3_1_plus = tf.add(res_3_1, res_3_3)
	res_3_4 = conv_block(res_3_1_plus, 256, kernel_size=[3, 3])
	res_3_5 = conv_block(res_3_4, 256, kernel_size=[3, 3])
	res_3_2_plus = tf.add(res_3_1_plus, res_3_5)

	up_2 = conv_transpose_block(res_3_2_plus, 128)
	plus_2 = tf.add(up_2, pool_2_1)
	res_2_1 = conv_block(plus_2, 128, kernel_size=[3, 3])
	res_2_2 = conv_block(res_2_1, 128, kernel_size=[3, 3])
	res_2_3 = conv_block(res_2_2, 128, kernel_size=[3, 3])
	res_2_1_plus = tf.add(res_2_1, res_2_3)
	res_2_4 = conv_block(res_2_1_plus, 128, kernel_size=[3, 3])
	res_2_5 = conv_block(res_2_4, 128, kernel_size=[3, 3])
	res_2_2_plus = tf.add(res_2_1_plus, res_2_5)

	up_1 = conv_transpose_block(res_2_2_plus, 64)
	plus_2 = tf.add(up_1, pool_1_1)
	res_1_1 = conv_block(plus_2, 64, kernel_size=[3, 3])
	res_1_2 = conv_block(res_1_1, 64, kernel_size=[3, 3])
	res_1_3 = conv_block(res_1_2, 64, kernel_size=[3, 3])
	res_1_1_plus = tf.add(res_1_1, res_1_3)
	res_1_4 = conv_block(res_1_1_plus, 64, kernel_size=[3, 3])
	res_1_5 = conv_block(res_1_4, 64, kernel_size=[3, 3])
	res_1_2_plus = tf.add(res_1_1_plus, res_1_5)

	up_0 = conv_transpose_block(res_1_2_plus, 64)
	plus_1 = tf.add(up_0, pool_0_2)
	res_0_1 = conv_block(plus_1, 64, kernel_size=[3, 3])
	res_0_2 = conv_block(res_0_1, 64, kernel_size=[3, 3])

	net = slim.conv2d(res_0_2, num_classes, [1,1], activation_fn = nn.sigmoid,scope = 'logits')
	return net