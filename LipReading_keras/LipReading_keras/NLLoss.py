import tensorflow as tf
from keras import backend as K

def NLLoss(y_true, y_pred):
    """ Negative log likelihood. """
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    y_true = tf.transpose(y_true, perm=[1, 0, 2])
    losses = 0
    for i in range(0,29):
        
        likelihood = K.tf.distributions.Bernoulli(probs=y_pred[i])
        losses += (- K.sum(likelihood.log_prob(y_true[i]), axis=-1))
    return losses