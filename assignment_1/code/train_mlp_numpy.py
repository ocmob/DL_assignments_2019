"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

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
  loss_module = CrossEntropyModule()

  weights_norms = []
  grad_norms = []

  weights_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))
  for i in range(len(dnn_hidden_units)):
      weights_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))

  grad_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))
  for i in range(len(dnn_hidden_units)):
      grad_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))

  loss_train = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  loss_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  accuracy_test = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))

  images_test = test.images
  labels_test = test.labels
  images_test = np.reshape(images_test, (images_test.shape[0], dim_x))

  for i in range(0, FLAGS.max_steps):
      print('iter', i+1, end='\r')
      images, labels = train.next_batch(FLAGS.batch_size) 
      images = np.reshape(images, (images.shape[0], dim_x))

      pred = mlp.forward(images)
      loss = loss_module.forward(pred, labels)
      loss_grad = loss_module.backward(pred, labels)
      mlp.backward(loss_grad)

      for module in reversed(mlp.modules):
          if isinstance(module, LinearModule):
              module.params['weight'] -= 1/FLAGS.batch_size*FLAGS.learning_rate*module.grads['weight']
              module.params['bias'] -= 1/FLAGS.batch_size*FLAGS.learning_rate*module.grads['bias']
      if (i+1) % FLAGS.eval_freq == 0:
          pred_test = mlp.forward(images_test)
          loss_train[i // FLAGS.eval_freq] = loss
          accuracy_test[i // FLAGS.eval_freq] = accuracy(pred_test, labels_test)
          loss_test[i // FLAGS.eval_freq] = loss_module.forward(pred_test, labels_test)

          cnt = 0
          for module in reversed(mlp.modules):
              if isinstance(module, LinearModule):
                  weights_norms[cnt][i // FLAGS.eval_freq] = module.params['weight'].sum()+module.params['bias'].sum()
                  cnt += 1

          cnt = 0
          for module in reversed(mlp.modules):
              if isinstance(module, LinearModule):
                  grad_norms[cnt][i // FLAGS.eval_freq] = module.grads['weight'].sum()+module.grads['bias'].sum()
                  cnt += 1

          print()
          print('test_loss:', loss_test[i // FLAGS.eval_freq])
          print('test_accuracy:', accuracy_test[i // FLAGS.eval_freq])
          print('train_loss:', loss_train[i // FLAGS.eval_freq])
  fig, ax = plt.subplots(1, 2)
  #ax[0].plot(loss_train, label='Loss, train')
  #ax[0].plot(loss_test, label='Loss, test')
  #ax[1].plot(accuracy_test, label='Accuracy, test')
  indices = np.arange(0, len(loss_test), 1)
  for i, array in enumerate(grad_norms):
      ax[0].scatter(indices, array, label='grad layer {}'.format(len(dnn_hidden_units)+1-i))
      ax[0].legend()
      print(array)
  for i, array in enumerate(weights_norms):
      ax[1].scatter(indices, array, label='weights layer {}'.format(len(dnn_hidden_units)+1-i))
      ax[1].legend()
      print(array)
  fig.tight_layout()
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
