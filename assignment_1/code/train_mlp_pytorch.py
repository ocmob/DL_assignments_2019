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
  predictions = predictions.clone().cpu().detach()
  targets = targets.clone().cpu().detach()
  pred = predictions.numpy()

  tg = np.zeros_like(pred)
  i1 = np.arange(0, len(tg), 1)

  tg[i1, targets.numpy()] += 1
  i2 = np.argmax(pred, axis = 1)
  accuracy = tg[i1, i2].sum()/tg.sum()
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
  
  if not torch.cuda.is_available():
      print("WARNING: CUDA DEVICE IS NOT AVAILABLE, WILL TRAIN ON CPU")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  data = cifar10_utils.get_cifar10(FLAGS.data_dir)

  train = data['train']
  test = data['test']

  dim_x = train.images.shape[1]*train.images.shape[2]*train.images.shape[3]

  if FLAGS.init_type == 1:
      init_name = 'xavier_uniform'
  elif FLAGS.init_type == 2:
      init_name = 'xavier_normal'
  elif FLAGS.init_type == 3:
      init_name = 'kaiming_uniform'
  elif FLAGS.init_type == 4:
      init_name = 'kaiming_normal'
  elif FLAGS.init_type == 5:
      init_name = 'orthogonal'
  else:
      init_name = 'Pytorch default'

  mlp = MLP(dim_x, dnn_hidden_units, train.labels.shape[1], neg_slope, FLAGS.init_type)
  mlp.to(device)

  criterion = nn.CrossEntropyLoss()

  if FLAGS.optim_type == 1:
      opt_name = 'SGD'
      optimizer = optim.SGD(mlp.parameters(), FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)
  elif FLAGS.optim_type == 2:
      opt_name = 'RMSprop'
      optimizer = optim.RMSprop(mlp.parameters(), FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)
  elif FLAGS.optim_type == 3:
      opt_name = 'Adagrad'
      optimizer = optim.Adagrad(mlp.parameters(), FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)
  else:
      opt_name = 'Adam'
      optimizer = optim.Adam(mlp.parameters(), FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience = 5)

  weights_norms = []
  grad_norms = []

  weights_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))
  for i in range(len(dnn_hidden_units)):
      weights_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))

  grad_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))
  for i in range(len(dnn_hidden_units)):
      grad_norms.append(np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), ))))

  loss_train = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  loss_cv = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))
  accuracy_cv = np.zeros((int(np.floor(FLAGS.max_steps/FLAGS.eval_freq), )))

  cv_samples = len(test.images)//2
  test_samples = len(test.images)-cv_samples

  images_cv_np, labels_cv_np = test.next_batch(cv_samples)
  images_test_np, labels_test_np = test.next_batch(test_samples)

  images_cv_np = np.reshape(images_cv_np, (images_test_np.shape[0], dim_x))
  images_test_np = np.reshape(images_test_np, (images_test_np.shape[0], dim_x))

  for i in range(0, FLAGS.max_steps):
      print('iter', i+1, end='\r')
      images_np, labels_np = train.next_batch(FLAGS.batch_size) 
      images_np = np.reshape(images_np, (images_np.shape[0], dim_x))

      images = torch.from_numpy(images_np).to(device)
      labels = torch.from_numpy(np.argmax(labels_np, axis = 1)).to(device)

      optimizer.zero_grad()

      pred = mlp(images)
      loss = criterion(pred, labels.long())
      loss.backward()
      optimizer.step()

      del images
      del labels

      if (i+1) % FLAGS.eval_freq == 0:
          loss_train[i // FLAGS.eval_freq] = loss.item()

          images_cv = torch.from_numpy(images_cv_np).to(device)
          labels_cv = torch.from_numpy(np.argmax(labels_cv_np, axis = 1)).to(device)

          with torch.no_grad():
              cnt = 0
              for module in mlp.module_list:
                  if isinstance(module, nn.Linear):
                      weights_norms[cnt][i // FLAGS.eval_freq] = module.weight.abs().sum().item()
                      cnt += 1

              cnt = 0
              for module in mlp.module_list:
                  if isinstance(module, nn.Linear):
                      grad_norms[cnt][i // FLAGS.eval_freq] = module.weight.grad.abs().sum().item()
                      cnt += 1
              pred_cv = mlp(images_cv)

          accuracy_cv[i // FLAGS.eval_freq] = accuracy(pred_cv, labels_cv).item()
          loss_cv[i // FLAGS.eval_freq] = criterion(pred_cv, labels_cv.long()).item()
          print()
          print('cv_loss:', loss_cv[i // FLAGS.eval_freq])
          print('cv_accuracy:', accuracy_cv[i // FLAGS.eval_freq])
          print('train_loss:', loss_train[i // FLAGS.eval_freq])
          scheduler.step(loss_cv[i // FLAGS.eval_freq])

  images_test = torch.from_numpy(images_test_np)
  labels_test = torch.from_numpy(np.argmax(labels_test_np, axis = 1))

  pred_test = mlp(images_test)
  accuracy_test = accuracy(pred_test, labels_test)

  fig, ax = plt.subplots(1, 2, figsize=(12,6))
  fig.suptitle('Training curves for Pytorch MLP. Final result: {:.4f} test accuracy\nParameters: Hidden units config: {}, Init type: {}, Optimizer type: {}, Weight decay: {} '.format(accuracy_test, FLAGS.dnn_hidden_units, init_name, opt_name, FLAGS.weight_decay))

  ax[0].set_title('Loss')
  ax[0].set_ylabel('Loss value')
  ax[0].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
  ax[0].plot(loss_train, label='Train')
  ax[0].plot(loss_cv, label='Cross-validation')
  ax[0].legend()

  ax[1].set_title('Accuracy')
  ax[1].set_ylabel('Accuracy value')
  ax[1].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
  ax[1].plot(accuracy_cv, label='Cross-validation')
  ax[1].legend()
  
  ## Lib, Nettype, Filetype, Init type, Steps, Batchsize, Eval_freq, Accuracy

  fig_name = 'pt_mlp_training_{}_{}_{}_{}_{:.4f}.jpg'.format(FLAGS.init_type, FLAGS.max_steps, FLAGS.batch_size, FLAGS.eval_freq, accuracy_test)
  plt.savefig(fig_name)

  fig, ax = plt.subplots(1, 2, figsize=(12,6))
  fig.suptitle('Norms of weight and gradient tensors for Pytorch MLP.'.format(accuracy_test, FLAGS.dnn_hidden_units, init_name, opt_name, FLAGS.weight_decay))
  indices = np.arange(0, len(loss_cv), 1)
  for i, array in enumerate(grad_norms):
      ax[0].scatter(indices, array, label='Layer {}'.format(len(dnn_hidden_units)+1-i))
  ax[0].set_title('Weight gradient matrix norms')
  ax[0].set_ylabel('Norm')
  ax[0].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
  ax[0].legend()
  for i, array in enumerate(weights_norms):
      ax[1].scatter(indices, array, label='Layer {}'.format(len(dnn_hidden_units)+1-i))
  ax[1].set_title('Weight matrix norms')
  ax[1].set_ylabel('Norm')
  ax[1].set_xlabel('No of batches seen x{}'.format(FLAGS.eval_freq))
  ax[1].legend()
  fig_name = 'pt_mlp_norms_{}_{}_{}_{}_{:.4f}.jpg'.format(FLAGS.init_type, FLAGS.max_steps, FLAGS.batch_size, FLAGS.eval_freq, accuracy_test)
  plt.savefig(fig_name)
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
  parser.add_argument('--init_type', type=float, default=0,
                      help='Type of initialization for linear layers')
  parser.add_argument('--optim_type', type=float, default=0,
                      help='Type of optimizer')
  parser.add_argument('--weight_decay', type=float, default=0.1,
                      help='Amount of weight decay')
  FLAGS, unparsed = parser.parse_known_args()

  main()
