from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data. (last column is a column of ones)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # score vector 1xC
        correct_class_score = scores[y[i]]
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                # each column in dw corresponding to the class which didn't meet the desired margin should merely be x_i
                dW[:, j] += X[i]
                # correct class should accumulate gradient term with each class which didn't meet the desired margin
                dW[:, y[i]] += -X[i]
        
        
    dW /= num_train
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    
    # don't regularize bias terms
    reg_term = W[:-1, :]
    reg_term = np.append(reg_term,[np.zeros(W.shape[1])],axis= 0)
    
    # Add regularization to the loss.
    loss += reg * np.sum(W*W) 
    
    # add regularization term to gradient
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data. (last column is a column of ones)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    n = X.shape[0]
    c = W.shape[1]
    
    s = np.matmul(X,W) # score matrix -> NxC
    correct_score = s[np.arange(n), y].reshape((n, 1)) 
    # get element wise max of s - correct_score before we take the sum of loss terms
    l_i = np.maximum(0.0, s - correct_score + 1) # delta = 1
    l_i[np.arange(n), y] = 0.0
    
    loss = np.sum(l_i)
    loss /= n
    
    # don't regularize bias terms
    reg_term = W[:-1, :]
    reg_term = np.append(reg_term,[np.zeros(W.shape[1])],axis= 0)
    
    # Add regularization to the loss.
    loss += 0.5*reg * np.sum(W*W)
    
    
    '''
    N = len(y)
    scores = X.dot(W) # N, C
    correct = scores[range(N),y].reshape((N,1))
    diffs = np.maximum(scores + 1.0 - correct, 0.0)
    diffs[range(N), y] = 0.0
    loss = diffs.sum() / N
    loss += 0.5 * reg * np.sum(W * W)
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    L = np.maximum(0, s - correct_score + 1) # delta = 1
    L[L>0] = 1
    L[np.arange(n), y] = -1
    missed_margins = np.sum(L>0, axis=1)
    L[np.arange(n), y] = -1*missed_margins

    dW = np.matmul(np.transpose(X), L)

    dW /= n
    dW += reg * W 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW