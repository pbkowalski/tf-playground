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
import matplotlib.pyplot as plt
import os.path
import random


#create the dataset
tf.reset_default_graph()


# Training Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 4
display_step = 100

# Network Parameters



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 68, 85, 1])
image_dir = "img"
if not tf.gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
extensions = ['jpg']
file_list = []
for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(tf.gfile.Glob(file_glob))
if not file_list:
    tf.logging.warning('No files found')    
    
#file_list = file_list[0:n]
#split into training and test
random.shuffle(file_list)
#proportion of data to be used for training
train = 0.9
filelist_train = file_list[:round(train*len(file_list))]
filelist_test = file_list[round(train*len(file_list)):]

filenamesTrain = tf.constant(filelist_train)
filenamesTest = tf.constant(filelist_test)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8)
    image = tf.cast(image_decoded, tf.float32)
    return image

datasetTrain = tf.data.Dataset.from_tensor_slices(filenamesTrain)
datasetTrain = datasetTrain.map(_parse_function)
datasetTrain = datasetTrain.repeat() #repeat the input indefinitely; num steps is used to control learning period
datasetTrain = datasetTrain.batch(batch_size)

#iterator = dataset.make_one_shot_iterator()
iterator = datasetTrain.make_one_shot_iterator()
#images = iterator.get_next()

datasetTest = tf.data.Dataset.from_tensor_slices(filenamesTest)
datasetTest = datasetTest.map(_parse_function)
datasetTest = datasetTest.repeat() #repeat the input indefinitely; num steps is used to control learning period
datasetTest = datasetTest.batch(batch_size)

#iterator = dataset.make_one_shot_iterator()
iteratorTest = datasetTest.make_one_shot_iterator()


def conv_encoder(x):
    #using 8x downsampling input is n*68*85*1
      conv1 = tf.layers.conv2d(
      inputs=x,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
     # 68 * 85 * 16
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      # 34 * 43 * 16
      # Convolutional Layer #2 and Pooling Layer #2
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=128,
          kernel_size=[2, 2],
          padding="same",
          activation=tf.nn.relu)
      # 34 * 43 * 16

      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      #17 * 22 * 16  
      return pool2
  
  
def conv_decoder(x):
      #upsample
      upsample1 = tf.image.resize_images(x, size=(34,43), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv3 = tf.layers.conv2d(inputs=upsample1, filters=128, kernel_size=(2,2), padding='same', activation=tf.nn.relu) 
      #34*43*16
      upsample2 = tf.image.resize_images(conv3, size=(68,85), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv4 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
      
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
    saver.restore(sess, "models/model2/model.cpkt")
    # Run the initializer
   # sess.run(init)
    
    logs_path = "logs"

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training


    print ("Model restored")    
    #quick and rubbish way of doing this - straight from the training dataset
    n=8 #test size
#    iterator = dataset.make_one_shot_iterator()
 #   imgs_it = iterator.get_next()
    canvas_orig = np.empty((68 * batch_size, 85 * n))
    canvas_recon = np.empty((68 * batch_size, 85 * n))
    
    imgs_itTest = iteratorTest.get_next();
    for i in range(n):
        imgs = sess.run(imgs_itTest)
      #  imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        g, loss = sess.run([decoder_op, loss], feed_dict={X: imgs})
        
        # Display original images
        for j in range(batch_size):
            # Draw the original digits
            canvas_orig[j * 68:(j + 1) * 68, i * 85:(i + 1) * 85] = \
                imgs[j].reshape([68, 85])
        # Display reconstructed images
        for j in range(batch_size):
            # Draw the reconstructed digits
            canvas_recon[j * 68:(j + 1) * 68, i * 85:(i + 1) * 85] = \
                g[j].reshape([68, 85])
    print("Original Images")
    plt.figure
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
   # plt.figure(figsize=(85/80*batch_size, 68/80*n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
    
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
