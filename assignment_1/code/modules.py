"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}

    self.params['weight'] = np.random.normal(0, 0.0001, size=(out_features, in_features))
    self.params['bias'] = np.zeros((out_features,))

    self.grads['weight'] = np.zeros((in_features, out_features)) # following numerator notation
    self.grads['bias'] = np.zeros((out_features,))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # out = B x D(ln)
    # x = B x D(ln-1)
    # W = D(ln) x D(ln-1)
    # B = D(ln)
    out = x @ self.params['weight'].T + self.params['bias']
    self.x = x.copy()
    #print('Forward, linear')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout @ self.params['weight']
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.sum(dout, axis=0)
    #print('Backward, linear')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.neg_slope = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x.copy()
    out = x*(x > 0) + self.neg_slope*x*(x <= 0)
    #print('Forward, lrelu')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx_chain = self.x
    dx_chain[dx_chain > 0] = 1
    dx_chain[dx_chain <= 0] = -self.neg_slope
    dx = dout*dx_chain
    #print('Backward, lrelu')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x_shift = np.exp(x - np.max(x, axis = 1, keepdims=True))
    out = x_shift/x_shift.sum(axis=1, keepdims=True)
    self.out = out.copy()
    #print('Forward, softmax')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    i1 = np.arange(0, len(dout), 1) 
    i2 = np.argmin(dout, axis=1) 
    dx = self.out
    dx[i1, i2] = dx[i1, i2] - 1
    #print('Backward, softmax')
    #breakpoint()

    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    i0 = np.arange(0, len(x), 1)
    i1 = np.argmax(y, axis = 1)
    out = -np.log(x[i0, i1])
    #print('Forward, Loss')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out.sum()

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = x*(y!=0)
    i0 = np.arange(0, len(dx), 1)
    i1 = np.argmax(dx, axis = 1)
    dx[i0, i1] = -1/dx[i0, i1]
    #print('Backward, Loss')
    #breakpoint()
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
