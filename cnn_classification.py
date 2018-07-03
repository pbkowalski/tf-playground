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
import pandas as pd

#create the dataset


# Training Parameters
learning_rate = 0.001
num_steps = 8000
batch_size = 16
display_step = 50
filters = 64
train = 0.8

# Network Parameters



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 68, 85, 1])
image_dir = "img3"
kasios_dir = "img4"

def _parse_folder(image_dir):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
    extensions = ['png', 'jpg']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')   
    return file_list

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

def _parse_function_nolabel(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image

def createDictionary(csv):
    df = pd.read_csv(csv)
    cols = df.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    df.columns = cols
    return dict(zip(list(df.File_ID), list(df.English_name)))

def parseFileList(filename):
    labelMap = createDictionary('AllBirdsv4.csv')
    foo = filename.split(os.sep)[-1].split('.')[0]
    val = int(foo)
    return labelMap[val]


file_list = _parse_folder(image_dir)
#file_list = file_list[0:n]
#split into training and test
random.shuffle(file_list)
labelList = list(map(parseFileList, file_list))
labels = list(set(labelList))
d = dict(zip(labels, range(0,len(labels))))
labelnList = list(map(lambda x : d[x], labelList))
#binary list
#val = d['Rose-crested Blue Pipit']
#labelnList = [1 if x==val else 0 for x in labelListnb]

##proportion of data to be used for training
filelist_train = file_list[:round(train*len(file_list))]
filelist_test = file_list[round(train*len(file_list)):]

labelList_train = labelnList[:round(train*len(file_list))]
labelList_test = labelnList[round(train*len(file_list)):]

filenamesTrain = tf.constant(filelist_train)
labelsTrain = tf.constant(labelList_train)
filenamesTest = tf.constant(filelist_test)
labelsTest = tf.constant(labelList_test)

kasiosList = _parse_folder(kasios_dir)



#images = iterator.get_next()

datasetTest = tf.data.Dataset.from_tensor_slices((filenamesTest, labelsTest))
datasetTest = datasetTest.map(_parse_function)
datasetTest = datasetTest.batch(1)
iteratorTest = datasetTest.make_one_shot_iterator()


  

# Construct model
def cnn_model(features, labels, mode):
      #input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
      #using 8x downsampling input is n*68*85*1
      conv1 = tf.layers.conv2d(
      inputs=features["x"],
      filters=filters,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
     # 68 * 85 * f
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

      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) #17 * 21 * 8
      conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=4*filters,
          kernel_size=[2, 2],
          padding="same",
          activation=tf.nn.relu)
      # 34 * 43 * 16

      pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
      #17 * 21 * 8

      num_input = 17*21*(filters*2)
      x_flat = tf.reshape(pool3, [-1, 20480])
      dense = tf.layers.dense(inputs=x_flat, units=1024, activation=tf.nn.relu)
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
      logits = tf.layers.dense(inputs=dropout, units=19)
    
      predictions = {
         # Generate predictions (for PREDICT and EVAL mode)
         "classes": tf.argmax(input=logits, axis=1),
         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
         # `logging_hook`.
         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
         }
         
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
      # Add evaluation metrics (for EVAL mode)
      confusion = tf.confusion_matrix(labels=labels, predictions=predictions["classes"], num_classes=19)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"]),
          "mean per class accuracy" : tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predictions["classes"], num_classes=19) }
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
# Create the Estimator
birdsclassifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir="/tmp/birds_cnn_model_2x")
 # Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)
print('begin training')
# Train the model
def input_fn():
    datasetTrain = tf.data.Dataset.from_tensor_slices((filenamesTrain, labelsTrain))
    datasetTrain = datasetTrain.map(_parse_function)
    datasetTrain = datasetTrain.repeat()  # repeat the input indefinitely;
    datasetTrain = datasetTrain.batch(batch_size)
    iteratorTrain = datasetTrain.make_one_shot_iterator()
    features, labels = iteratorTrain.get_next()
    return {'x': features}, labels
birdsclassifier.train(
    input_fn=input_fn,
    #steps=num_steps,
    steps=num_steps,
    hooks=[logging_hook])
# Evaluate the model and print results
print('begin eval')
def eval_input_fn():
    datasetTest = tf.data.Dataset.from_tensor_slices((filenamesTest, labelsTest))
    datasetTest = datasetTest.map(_parse_function)
    datasetTest = datasetTest.batch(1)
    iteratorTest = datasetTest.make_one_shot_iterator()
    featuresTest, labelsT= iteratorTest.get_next()
    return {'x': featuresTest}, labelsT

eval_results = birdsclassifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


#to compute confusion matrix

labels = labelsTest

predictions = [entry["classes"] for entry in birdsclassifier.predict(input_fn=eval_input_fn)]
confusion_matrix = tf.confusion_matrix(labels, predictions)
sess = tf.Session()
with sess.as_default():
    print(confusion_matrix.eval())
print('begin predict')
def predict_input_fn():
    kasiosFilenames = tf.constant(kasiosList)
    datasetKasios = tf.data.Dataset.from_tensor_slices(kasiosFilenames)
    datasetKasios = datasetKasios.map(_parse_function_nolabel)
    datasetKasios = datasetKasios.batch(1)
    kasiosIt = datasetKasios.make_one_shot_iterator()
    featuresKasios = kasiosIt.get_next()
    return {'x': featuresKasios}
predict_results = birdsclassifier.predict(input_fn=predict_input_fn)
print(list(predict_results))
print(d)