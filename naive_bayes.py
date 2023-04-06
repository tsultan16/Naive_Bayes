'''
    Implementation of a Naive Bayes Classifier. The target attribute/class label is assumed to be of cardinal type by default.
'''

import numpy as np

# to train the model, we need to compute all the prior probabilities of each class and the conditional probabilities of each attribute given each class.
def train_naive_bayes(instances, attributes, target):


    # compute class prior probabilities
    y = [instance[-1] for instance in instances]
    y_vals, counts = np.unique(y, return_counts=True)

    Py = {}
    N = len(instances)
    for i, y in enumerate(y_vals):
        Py[y] = counts[i]/N
    #print(f"Prior class probabilities: {Py}")

    Pxy = {}
    for i, attribute in enumerate([attribute for attribute in attributes if attribute is not target]):
        # compute gaussian distribution parameters for attribute conditional probabilities
        if(attributes[attribute] == 'continuous'):
            Pxy[attribute] = {}
            for y in y_vals:
                Pxy[attribute][y] = {}
                attribute_vals = [instance[i] for instance in instances if instance[-1] == y]
                 # compute sample mean and variance of attribute values of instances belonging to this class
                Pxy[attribute][y]['mean'] = sum(attribute_vals)/len(attribute_vals)
                Pxy[attribute][y]['variance'] = np.sqrt(sum([(val - Pxy[attribute][y]['mean'])**2 for val in attribute_vals])/len(attribute_vals))

            #print(Pxy[attribute])                

        elif(attributes[attribute] == 'cardinal'):
            print("Cardinal attributes not yet implemented!")
            return    


    return Py, Pxy


# Sample training data: Each instance has 3 continuous attributes X1, X2, and X3 and the binary target attribute/class Y.
attributes = {'X1' : 'continuous', 'X2' : 'continuous', 'X3' : 'continuous', 'Y' : 'cardinal'}
target_attribute = 'Y'
training_instances = [ [0.8, 0.4, 39.5, 'flu' ],
                       [0.0, 0.8, 37.8, 'cold'],
                       [0.4, 0.4, 37.8, 'flu' ],
                       [0.4, 0.0, 37.8, 'cold'],
                       [0.8, 0.8, 37.8, 'flu' ] ]


Py, Pxy = train_naive_bayes(training_instances, attributes, target_attribute)

print(f"Class prior probabilities: {Py}")
print(f"Conditional probability gaussian paramenets: {Pxy}")
