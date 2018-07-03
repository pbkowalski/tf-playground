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


#create the dataset
tf.reset_default_graph()


# Training Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 1
display_step = 100
filters = 2

# Network Parameters



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 68, 85, 1])
image_dir = "img3"
if not tf.gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
extensions = ['jpg']
file_list = []
for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(tf.gfile.Glob(file_glob))
if not file_list:
    tf.logging.warning('No files found')    
    
filenames = tf.constant(file_list)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8)
    image = tf.cast(image_decoded, tf.float32)
    return image

dataset = tf.data.Dataset.from_tensor_slices(filenames)

dataset = dataset.map(_parse_function)
dataset = dataset.batch(1)

#iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_one_shot_iterator()



def conv_encoder(x):
    #using 8x downsampling input is n*68*85*1
      conv1 = tf.layers.conv2d(
      inputs=x,
      filters=filters,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
     # 68 * 85 * 64
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      # 34 * 43 * 16
      # Convolutional Layer #2 and Pooling Layer #2
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=filters*2,
          kernel_size=[2, 2],
          padding="same",
          activation=tf.nn.relu)
      # 34 * 43 * 128

      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      #17 * 22 * 16  
      return pool2
  
  
def conv_decoder(x):
      #upsample
      upsample1 = tf.image.resize_images(x, size=(34,43), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv3 = tf.layers.conv2d(inputs=upsample1, filters=filters*2, kernel_size=(2,2), padding='same', activation=tf.nn.relu) 
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
saver = tf.train.Saver()
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session

with tf.Session() as sess:
#    saver = tf.train.import_meta_graph("models/model2/model.cpkt.meta")
    saver.restore(sess, "models/model4/model.cpkt")
    # Run the initializer
   # sess.run(init)
    
    logs_path = "logs"

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training


    print ("Model restored")    
    #compute loss for every element

    
    imgs_it = iterator.get_next();
    lossList = []
    while True:
        try:
            imgs = sess.run(imgs_it)
#            g = sess.run(decoder_op, feed_dict={X: imgs} )
            l = sess.run(loss, feed_dict={X: imgs} )
            lossList.append(l)


        except tf.errors.OutOfRangeError:
            break

f = open("output6_test2.csv", "w")
for i in range(len(file_list)):
    f.write("{}, {}\n".format(file_list[i], lossList[i]))
f.close()
nploss = np.array(lossList)    
plt.hist(nploss, bins = 'auto')
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
