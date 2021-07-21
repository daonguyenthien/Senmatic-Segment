import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import glob
from utils import utils, helpers
from builders import model_builder
import time
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 3
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='./output/', required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/0200/Face_seg_model/latest_model_Encoder-Decoder-Skip_Face_seg.ckpt', required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default='Encoder-Decoder-Skip', required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="Face_seg/", required=False, help='The dataset you are using')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)



fig.add_subplot(rows, columns, 1)
replace_img_1 = cv2.cvtColor(cv2.imread('./Images/1_V1.png'),cv2.COLOR_BGR2RGB)
img_1 = replace_img_1
plt.imshow(replace_img_1)
plt.axis('off')
plt.title("ori_1")


resized_image_1 =cv2.resize(replace_img_1, (args.crop_width, args.crop_height))
input_image_1 = np.expand_dims(np.float32(resized_image_1[:args.crop_height, :args.crop_width]),axis=0)/255.0
output_image_1 = sess.run(network,feed_dict={net_input:input_image_1})
output_image_1 = np.array(output_image_1[0,:,:,:])
output_image_1 = helpers.reverse_one_hot(output_image_1)
out_vis_image_1 = helpers.colour_code_segmentation(output_image_1, label_values)

fig.add_subplot(rows, columns, 2)
plt.imshow(out_vis_image_1)
plt.axis('off')
plt.title("pred_1")



fig.add_subplot(rows, columns, 4)
replace_img_2 = cv2.cvtColor(cv2.imread('./Images/3_V1.png'),cv2.COLOR_BGR2RGB)
img_2 = replace_img_2
mask = np.zeros_like(img_2)
plt.imshow(replace_img_2)
plt.axis('off')
plt.title("ori_2")


resized_image_2 =cv2.resize(replace_img_2, (args.crop_width, args.crop_height))
input_image_2 = np.expand_dims(np.float32(resized_image_2[:args.crop_height, :args.crop_width]),axis=0)/255.0
output_image_2 = sess.run(network,feed_dict={net_input:input_image_2})
output_image_2 = np.array(output_image_2[0,:,:,:])
output_image_2 = helpers.reverse_one_hot(output_image_2)
print()
out_vis_image_2 = helpers.colour_code_segmentation(output_image_2, label_values)

fig.add_subplot(rows, columns, 5)
plt.imshow(out_vis_image_2)
plt.axis('off')
plt.title("pred_2")
landmarks_points = []
for x in range(img_2.shape[0]):
    for y in range(img_2.shape[1]):
        #print(out_vis_image_2[x,y])
        b_1 , g_1 , r_1 = out_vis_image_2[x,y]
        b_2 , g_2 , r_2 = out_vis_image_1[x,y]
        if (b_1 , g_1 , r_1 ) == (0,255,0):
            landmarks_points.append((x,y))
points = np.array(landmarks_points, np.int32)
convexhull = cv2.convexHull(points)
cv2.fillConvexPoly(mask, convexhull, 255)
face_image_1 = cv2.bitwise_and(replace_img_2, replace_img_2, mask=mask)
fig.add_subplot(rows, columns, 3)
plt.imshow(face_image_1)
plt.axis('off')
plt.title("pred_2")
plt.show()
'''   
       #find and replace part
        for i in range(len(color_to_seek)):
            
            
                    b, g, r = replace_img[x, y]
                    b1, g1, r1 = blend_img[x,y]
                    if (b, g, r) == (b1,g1,r1):
                        # thay doi mat muoi mieng ben hinh replace thanh cua minh
                        resized_image[x,y] = replace_img_1[x,y]
'''
