""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import random
import os
from os.path import isfile, join # to manupulate file paths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#create the dataset
output_dir= 'autoencoder_out_pipits_2'


# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 4
display_step = 50
filters = 32
train = 0.914

# Network Parameters



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 68, 85, 1])
image_dir = "img_pipit_filtered"


def _parse_folder(image_dir):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
    extensions = ['jpg']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')   
    return file_list

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8, channels = 1)
    image = tf.cast(image_decoded, tf.float32)
    return image


    
file_list = _parse_folder(image_dir)
#file_list = file_list[0:n]
#split into training and test
random.shuffle(file_list)
#proportion of data to be used for training
filelist_train = file_list[:round(train*len(file_list))]
filelist_test = file_list[round(train*len(file_list)):]

filenamesTrain = tf.constant(filelist_train)
filenamesTest = tf.constant(filelist_test)

kasiosList = _parse_folder("img2")
kasiosFilenames = tf.constant(kasiosList)
datasetKasios = tf.data.Dataset.from_tensor_slices(kasiosFilenames)
datasetKasios = datasetKasios.map(_parse_function)
datasetKasios = datasetKasios.batch(1)
kasiosIt = datasetKasios.make_one_shot_iterator()

datasetTrain = tf.data.Dataset.from_tensor_slices(filenamesTrain)
datasetTrain = datasetTrain.map(_parse_function)
datasetTrain = datasetTrain.repeat() #repeat the input indefinitely; num steps is used to control learning period
datasetTrain = datasetTrain.batch(batch_size)

#iterator = dataset.make_one_shot_iterator()
iterator = datasetTrain.make_one_shot_iterator()
#images = iterator.get_next()

datasetTest = tf.data.Dataset.from_tensor_slices(filenamesTest)
datasetTest = datasetTest.map(_parse_function)
datasetTest = datasetTest.batch(1)

iteratorTest = datasetTest.make_one_shot_iterator()


def conv_encoder(x):
    #using 8x downsampling input is n*68*85*1
      conv1 = tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
     # 68 * 85 * 16
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      # 34 * 43 * 16
      # Convolutional Layer #2 and Pooling Layer #2
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=2*filters,
          kernel_size=[2, 2],
          padding="same",
          activation=tf.nn.relu)
      # 34 * 43 * 16

      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      #17 * 21 * 8 
      return pool2
  
  
def conv_decoder(x):
      #upsample
      upsample1 = tf.image.resize_images(x, size=(34,43), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv3 = tf.layers.conv2d(inputs=upsample1, filters=2*filters, kernel_size=(2,2), padding='same', activation=tf.nn.relu) 
      #34*43*16
      upsample2 = tf.image.resize_images(conv3, size=(68,85), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv4 = tf.layers.conv2d(inputs=upsample2, filters=filters, kernel_size=(2,2), padding='same', activation=tf.nn.relu)
      
      logits = tf.layers.conv2d(inputs=conv4, filters=1, kernel_size=(2,2), padding='same', activation=None)
      #Now 68*85*1
     # decoded = tf.nn.sigmoid(logits)
      return logits

# Construct model
encoder_op = conv_encoder(X)
#encoder_op = encoder(images)

decoder_op = conv_decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.

y_true = X
#y_true = images

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
saver = tf.train.Saver()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    logs_path = "logs"

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    imgs_it = iterator.get_next();
    # Training


    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
       # batch_x, _ = mnist.train.next_batch(batch_size)
        
     #   print(imgs.get_shape())
        #except tf.errors.OutOfRangeError:
         #   print("error")
          #  break
        # Run optimization op (backprop) and cost op (to get loss value)
        imgs = sess.run(imgs_it)
     #   imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        _, l = sess.run([optimizer, loss], feed_dict={X: imgs})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
        
    #quick and rubbish way of doing this - straight from the training dataset
    n=2 #test size
#    iterator = dataset.make_one_shot_iterator()
 #   imgs_it = iterator.get_next()
    canvas_orig = np.empty((68 * 1, 85 * n))
    canvas_recon = np.empty((68 * 1, 85 * n))
    
    imgs_itTest = iteratorTest.get_next();
    for i in range(n):
        imgs = sess.run(imgs_itTest)
      #  imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        g = sess.run(decoder_op, feed_dict={X: imgs})
        l = sess.run(loss, feed_dict={X: imgs} )
        print(l)
        # Display original images
        for j in range(1):
            # Draw the original digits
            canvas_orig[j * 68:(j + 1) * 68, i * 85:(i + 1) * 85] = \
                imgs[j].reshape([68, 85])
        # Display reconstructed images
        for j in range(1):
            # Draw the reconstructed digits
            canvas_recon[j * 68:(j + 1) * 68, i * 85:(i + 1) * 85] = \
                g[j].reshape([68, 85])
    saver.save(sess, "models/model4/model.cpkt")
    lossList = []
    while True:
        try:
            imgs = sess.run(imgs_itTest)
#            g = sess.run(decoder_op, feed_dict={X: imgs} )
            l = sess.run(loss, feed_dict={X: imgs} )
            lossList.append(l)
        except tf.errors.OutOfRangeError:
            break
    imgsKasios = kasiosIt.get_next()
    lossListKasios = []
    while True:
        try:
            imgs = sess.run(imgsKasios)
#            g = sess.run(decoder_op, feed_dict={X: imgs} )
            l = sess.run(loss, feed_dict={X: imgs} )
            lossListKasios.append(l)
        except tf.errors.OutOfRangeError:
            break    
        
# print("Original Images")
# plt.figure(num = 1)
# plt.imshow(canvas_orig, origin="upper", cmap="gray")
# plt.show()
#
# print("Reconstructed Images")
# plt.figure(num = 2)
#    # plt.figure(figsize=(85/80*batch_size, 68/80*n))
# plt.imshow(canvas_recon, origin="upper", cmap="gray")
# plt.show()
#


output_prefix_1 = 'ref_eval_loss_'
output_prefix_2 = 'kassios_loss_'
output_prefix_3 = 'histogram'
it = 1
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename1 = output_prefix_1 + str(it) + '.csv'


while isfile(join(output_dir, output_filename1)):
    it += 1
    output_filename1 = output_prefix_1 + str(it) + '.csv'

output_filename2 = output_prefix_2 + str(it) + '.csv'

f = open(join(output_dir, output_filename1), "w")
for i in range(len(lossList)):
    f.write("{}\n".format(lossList[i]))
f.close()

f = open(join(output_dir, output_filename2), "w")
for i in range(len(lossListKasios)):
    f.write("{}\n".format(lossListKasios[i]))
f.close()

output_filename3 = output_prefix_3 + str(it) + '.png'

print("Histogram - remaining test images")
plt.figure(num = 3)
nploss = np.array(lossList)
plt.hist(nploss, bins = 'auto', alpha = 0.5, label = 'reference images (evaluation set)')
nploss2 = np.array(lossListKasios)
plt.hist(nploss2, bins = 'auto', alpha = 0.5, label = 'kasios images')
plt.legend(loc='upper right')

plt.savefig(join(output_dir,output_filename3))
# Testing
    # Encode and decode images from test set and visualize their reconstruction.
