"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.module_list = nn.ModuleList()
    self.module_list.append((nn.Conv2d(3, 64, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(64))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.MaxPool2d(3, 2, 1))
    self.module_list.append(nn.Conv2d(64, 128, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(128))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.MaxPool2d(3, 2, 1))
    self.module_list.append(nn.Conv2d(128, 256, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(256))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.Conv2d(256, 256, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(256))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.MaxPool2d(3, 2, 1))
    self.module_list.append(nn.Conv2d(256, 512, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(512))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.Conv2d(512, 512, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(512))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.MaxPool2d(3, 2, 1))
    self.module_list.append(nn.Conv2d(512, 512, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(512))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.Conv2d(512, 512, 3, padding=1))
    self.module_list.append(torch.nn.BatchNorm2d(512))
    self.module_list.append(torch.nn.ReLU())

    self.module_list.append(nn.MaxPool2d(3, 2, 1))
    self.module_list.append(nn.Linear(512, 10))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
