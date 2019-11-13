"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  import matplotlib.pyplot as plt

  data = cifar10_utils.get_cifar10(FLAGS.data_dir)

  train = data['train']
  test = data['test']

  vgg = ConvNet(3, 10)

  criterion = nn.CrossEntropyLoss()
  #TODO: ADAM?
  optimizer = optim.SGD(vgg.parameters(), FLAGS.learning_rate)

  loss_train = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  loss_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  accuracy_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))

  images_test_np = test.images
  labels_test_np = test.labels

  images_test = torch.from_numpy(images_test_np)
  labels_test = torch.from_numpy(np.argmax(labels_test_np, axis = 1))

  for i in range(0, FLAGS.max_steps):
      print('iter', i+1, end='\r')
      images_np, labels_np = train.next_batch(FLAGS.batch_size) 

      images = torch.from_numpy(images_np)
      labels = torch.from_numpy(np.argmax(labels_np, axis = 1))

      optimizer.zero_grad()

      pred = vgg(images)
      loss = criterion(pred, labels.long())
      loss.backward()
      optimizer.step()

      if (i+1) % FLAGS.eval_freq == 0:
          loss_train[i // FLAGS.eval_freq] = loss.item()
          pred_test = vgg(images_test)
          accuracy_test[i // FLAGS.eval_freq] = accuracy(pred_test, labels_test)
          loss_test[i // FLAGS.eval_freq] = criterion(pred_test, labels_test.long()).item()
          print()
          print('test_loss:', loss_test[i // FLAGS.eval_freq])
          print('test_accuracy:', accuracy_test[i // FLAGS.eval_freq])
          print('train_loss:', loss_train[i // FLAGS.eval_freq])
  fig, ax = plt.subplots(1, 2)
  ax[0].plot(loss_train, label='Loss, train')
  ax[0].plot(loss_test, label='Loss, test')
  ax[1].plot(accuracy_test, label='Accuracy, test')
  fig.legend()
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
