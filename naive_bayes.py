'''
    Implementation of a Naive Bayes Classifier. The target attribute/class label is assumed to be of cardinal type by default.
'''

import numpy as np

# to train the model, we need to compute all the prior probabilities of each class and the conditional probabilities of each attribute given each class.
def train_naive_bayes(instances, attributes):

    print("Cardinal attributes not yet implemented!")

    # compute class prior probabilities
    y = [instance[-1] for instance in instances]
    y_vals, counts = np.unique(y, return_counts=True)
    y = {y_val:count for y_val, count in zip(y_vals, counts)}

    Py = {}
    N = len(instances)
    for y_val in y:
        Py[y_val] = y[y_val]/N






# Sample training data: Each instance has 3 continuous attributes X1, X2, and X3 and the binary target attribute/class Y.
attributes = {'X1' : 'continuous', 'X2' : 'continuous', 'X3' : 'continuous', 'Y' : 'cardinal'}
training_instances = [ [0.8, 0.4, 39.5, 'flu' ],
                       [0.0, 0.8, 37.8, 'cold'],
                       [0.4, 0.4, 37.8, 'flu' ],
                       [0.4, 0.0, 37.8, 'cold'],
                       [0.8, 0.8, 37.8, 'flu' ] ]


train_naive_bayes(training_instances, attributes)
