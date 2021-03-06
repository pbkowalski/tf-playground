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


# Training Parameters
learning_rate = 0.5
num_steps = 1000
batch_size = 1
display_step = 100

# Network Parameters
num_hidden_1 = 4000 # 1st layer num features
num_hidden_2 = 1000 # 1st layer num features
#num_hidden_3 = 1024 # 2nd layer num features (the latent dim)
#num_input = 369920 # MNIST data input (img shape: 680*544) 
#num_input = 23120 # MNIST data input (img shape: 680 x 544 downscalred by ratio 4)
num_input = 5780 # MNIST data input (img shape: 680 x 544 downscalred by ratio 8)
n=3 #reduced dataset size


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
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
    
file_list = file_list[0:n]
filenames = tf.constant(file_list)
dataset = tf.data.Dataset.from_tensor_slices(filenames)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8)
    image = tf.cast(image_decoded, tf.float32)
    return image

dataset = dataset.map(_parse_function)
dataset = dataset.repeat() #repeat the input indefinitely; num steps is used to control learning period
dataset = dataset.batch(batch_size)

#iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_one_shot_iterator()
#images = iterator.get_next()



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
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
  #  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
  #                                 biases['encoder_b3']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
 #   layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
 #                                  biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   biases['decoder_b2']))
 #   # Decoder Hidden layer with sigmoid activation #2
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                  biases['decoder_b3']))
    return layer_3

# Construct model
encoder_op = encoder(X)
#encoder_op = encoder(images)

decoder_op = decoder(encoder_op)

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
        imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        _, l = sess.run([optimizer, loss], feed_dict={X: imgs})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
        
    #quick and rubbish way of doing this - straight from the training dataset

#    iterator = dataset.make_one_shot_iterator()
 #   imgs_it = iterator.get_next()
    canvas_orig = np.empty((68 * batch_size, 85 * n))
    canvas_recon = np.empty((68 * batch_size, 85 * n))
    for i in range(n):
        imgs = sess.run(imgs_it)
        imgs = np.reshape(imgs, (imgs.shape[0], num_input))
        g = sess.run(decoder_op, feed_dict={X: imgs})
        
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
#    saver.save(sess, "models/model2/model.cpkt")

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
