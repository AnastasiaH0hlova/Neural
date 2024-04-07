import numpy as np

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
    - loss as a single float
    - gradient with respect to weights W; an array of the same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_samples = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_samples):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # for numeric stability
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores)
        probs = exp_scores / sum_exp_scores
        correct_class_prob = probs[y[i]]
        loss += -np.log(correct_class_prob)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    # Average the loss and gradients over all examples
    loss /= num_samples
    dW /= num_samples

    # Add regularization to the loss and gradients
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_samples = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # for numeric stability
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sum_exp_scores
    correct_class_probs = probs[range(num_samples), y]

    # Compute the loss
    loss = -np.sum(np.log(correct_class_probs))
    loss /= num_samples
    loss += 0.5 * reg * np.sum(W * W)

    # Compute the gradient
    dscores = probs
    dscores[range(num_samples), y] -= 1
    dscores /= num_samples

    dW = X.T.dot(dscores)
    dW += reg * W

    return loss, dW