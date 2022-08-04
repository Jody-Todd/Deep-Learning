from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    num_classes = W.shape[1]

    for i in range(n):
        # class score for Ni
        fi = np.matmul(X[i], W) # 1xC
        
        # correct for numeric instability by mulitplying numerator and denominator by c, where log(c)= -max(fi). this asserts that 
        # the highest value in fi is 0.
        fi -= np.max(fi)
        
        fyi = fi[y[i]]
        loss += -(fyi)
        loss += np.log(np.sum(np.exp(fi)))
        

        for j in range(num_classes):
            # get gradient using chain rule
            dW[:,j] += ( (np.exp(fi[j]) * X[i]) / (np.sum(np.exp(fi))) )
        dW[:, y[i]] -= X[i]
        
        
        
    dW /= n
    dW += 2*reg*W
    
    loss /= n
    loss += 0.5*reg*np.sum(W*W)
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) # NxD, D,C -> NxC
    # stabilize
    scores -= np.max(scores, axis=1, keepdims=True)
    sum = np.sum(np.exp(scores), axis = 1)  # Nx1
    log = np.log(sum)
    loss = log - scores[np.arange(n), y]
    loss = np.sum(loss)
    loss /= n
    # regularization
    loss += 0.5*reg*np.sum(W*W)
    
    ind = np.zeros_like(scores) # NxC
    ind[np.arange(n), y] = 1 
    dW[:, :] = X.T.dot( (np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) ) - ind) # DxN, (NxC - NxC) -> DxC

    dW /= n
    dW += 2*reg*W
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
