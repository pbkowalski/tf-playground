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
# Training Parameters
learning_rate = 0.01
learning_rate_fnn = 0.001

num_steps = 500
num_steps_fnn = 1000

batch_size = 1
display_step = 100


#create the dataset and iterators

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
    
#file_list = file_list[0:n]
#split into training and test
random.shuffle(file_list)
#proportion of data to be used for training
train = 0.5
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





# Network Parameters



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 68, 85, 1])
cnn_filters = 3

#input to the FNN autoencoder (output of the CNN encoder)
num_input = 17*21*(cnn_filters*2)
num_hidden_1 =  1714# 1st layer num features
num_hidden_2 = 1200 # 2nd layer num features
Y = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),

    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),

    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def fnn_encoder(x):
    x_flat = tf.reshape(x, [-1, num_input])

    # Encoder Hidden layer with sigmoid activation #1
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_flat, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
  #  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
  #                                 biases['encoder_b3']))
    return layer_2


# Building the decoder
def fnn_decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
 #   layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
 #                                  biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   biases['decoder_b2']))
 #   # Decoder Hidden layer with sigmoid activation #2
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                  biases['decoder_b3']))
    layer_3_reshaped = tf.reshape(layer_3, [-1,17,21,cnn_filters*2])
    return layer_3_reshaped


def conv_encoder(x):
    #using 8x downsampling input is n*68*85*1
      conv1 = tf.layers.conv2d(
      inputs=x,
      filters=cnn_filters,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
     # 68 * 85 * 16
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      # 34 * 43 * 16
      # Convolutional Layer #2 and Pooling Layer #2
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=cnn_filters*2,
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
      
      conv3 = tf.layers.conv2d(inputs=upsample1, filters=cnn_filters*2, kernel_size=(2,2), padding='same', activation=tf.nn.relu) 
      #34*43*16
      upsample2 = tf.image.resize_images(conv3, size=(68,85), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      conv4 = tf.layers.conv2d(inputs=upsample2, filters=cnn_filters, kernel_size=(2,2), padding='same', activation=tf.nn.relu)
      
      logits = tf.layers.conv2d(inputs=conv4, filters=1, kernel_size=(2,2), padding='same', activation=None)
      #Now 68*85*1
     # decoded = tf.nn.sigmoid(logits)
      return logits

# Construct model
conv_encoder_op = conv_encoder(X)
#encoder_op = encoder(images)

conv_decoder_op = conv_decoder(conv_encoder_op)

fnn_encoder_op = fnn_encoder(conv_encoder_op)
fnn_decoder_op = fnn_decoder(fnn_encoder_op)


#Model 2 - full model

cnn_enc2 = conv_encoder(X)
fnn_enc2 = fnn_encoder(cnn_enc2)
fnn_dec2 = fnn_decoder(fnn_enc2)
cnn_dec2 = conv_decoder(fnn_dec2)

# Prediction
y_pred = conv_decoder_op #cnn autoencoder prediction

#create a list of all the varaibles associated with the fnn autoencoder only
fnn_vars = []
fnn_vars.extend(weights.values())
fnn_vars.extend(biases.values())


all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

fnn_loss = tf.reduce_mean(tf.pow(conv_encoder_op - fnn_decoder_op, 2))
total_loss = tf.reduce_mean(tf.pow(y_true - cnn_dec2, 2))

fnn_optimizer = tf.train.AdamOptimizer(learning_rate_fnn).minimize(fnn_loss, var_list = fnn_vars) #optimize fnn separately from cnn
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) #cnn optimization



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
    # Training CNN
    for i in range(1, num_steps+1):
        # Run optimization op (backprop) and cost op (to get loss value)
        imgs = sess.run(imgs_it)
     #   imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        _, l = sess.run([optimizer, loss], feed_dict={X: imgs})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
    #Training FNN
    for i in range(1, num_steps_fnn+1):
        # Run optimization op (backprop) and cost op (to get loss value)
        imgs = sess.run(imgs_it)
     #   imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        _, l = sess.run([fnn_optimizer, total_loss], feed_dict={X: imgs})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss(FNN): %f' % (i, l))       
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
        g = sess.run(cnn_dec2, feed_dict={X: imgs})
        
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
 #   saver.save(sess, "models/model4/model.cpkt")

print("Original Images")
plt.figure
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
   # plt.figure(figsize=(85/80*batch_size, 68/80*n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

f = open("trainfilelist.csv", "w")
for i in range(len(filelist_train)):
    f.write("{}\n".format(filelist_train[i]))
f.close()
# Testing
    # Encode and decode images from test set and visualize their reconstruction.
