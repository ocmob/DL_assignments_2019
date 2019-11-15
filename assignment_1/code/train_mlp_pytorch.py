"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

PRINTS = False
PLOTS = False

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
  predictions = predictions.clone().detach().numpy()
  targets = targets.clone().detach().numpy()

  i1 = np.arange(0, len(targets), 1)
  i2 = np.argmax(predictions, axis = 1)
  accuracy = targets[i1, i2].sum()/targets.sum()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  import matplotlib.pyplot as plt

  data = cifar10_utils.get_cifar10(FLAGS.data_dir)

  train = data['train']
  test = data['test']

  dim_x = train.images.shape[1]*train.images.shape[2]*train.images.shape[3]

  mlp = MLP(dim_x, dnn_hidden_units, train.labels.shape[1], neg_slope)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(mlp.parameters(), FLAGS.learning_rate)

  loss_train = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  loss_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  accuracy_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))

  images_test_np = test.images
  labels_test_np = test.labels
  images_test_np = np.reshape(images_test_np, (images_test_np.shape[0], dim_x))

  images_test = torch.from_numpy(images_test_np)
  labels_test = torch.from_numpy(np.argmax(labels_test_np, axis = 1))

  for i in range(0, FLAGS.max_steps):
      if PRINTS:
          print('iter', i+1, end='\r')

      images_np, labels_np = train.next_batch(FLAGS.batch_size) 
      images_np = np.reshape(images_np, (images_np.shape[0], dim_x))

      images = torch.from_numpy(images_np)
      labels = torch.from_numpy(np.argmax(labels_np, axis = 1))

      optimizer.zero_grad()

      pred = mlp(images)
      loss = criterion(pred, labels.long())
      loss.backward()
      optimizer.step()

      if (i+1) % FLAGS.eval_freq == 0:
          loss_train[i // FLAGS.eval_freq] = loss.item()
          pred_test = mlp(images_test)
          accuracy_test[i // FLAGS.eval_freq] = accuracy(pred_test, F.one_hot(labels_test))
          loss_test[i // FLAGS.eval_freq] = criterion(pred_test, labels_test.long()).item()
          if PRINTS:
              print()
              print('test_loss:', loss_test[i // FLAGS.eval_freq])
              print('test_accuracy:', accuracy_test[i // FLAGS.eval_freq])
              print('train_loss:', loss_train[i // FLAGS.eval_freq])
  if PLOTS:
      fig, ax = plt.subplots(1, 2, figsize=(10,5))
      fig.suptitle('Training curves for Pytorch MLP')

      ax[0].set_title('Loss')
      ax[0].set_ylabel('Loss value')
      ax[0].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
      ax[0].plot(loss_train, label='Train')
      ax[0].plot(loss_test, label='Test')
      ax[0].legend()

      ax[1].set_title('Accuracy')
      ax[1].set_ylabel('Accuracy value')
      ax[1].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
      ax[1].plot(accuracy_test, label='Test')
      ax[1].legend()
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()
