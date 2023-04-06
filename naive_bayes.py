'''
    Implementation of a Naive Bayes Classifier. The target attribute/class label is assumed to be of cardinal type by default.
'''

import numpy as np

# to train the model, we need to compute all the prior probabilities of each class and the conditional probabilities of each attribute given each class.
def train_naive_bayes(instances, attributes):

    # compute class prior probabilities
    y_vals, counts = get_y_vals(instances)

    Py = {}
    N = len(instances)
    for i, y in enumerate(y_vals):
        Py[y] = counts[i]/N
    #print(f"Prior class probabilities: {Py}")

    Pxy = {}
    for i, attribute in enumerate([attribute for attribute in attributes if attribute is not list(attributes.keys())[-1]]):
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

# function for clasifying a given test instance using naive bayes model 
def predict_class(X, Py, Pxy_params, attributes, target_attribute_vals):

    print(f"Test instance: {X}")

    # compute posterior probabilities of attributes given each class
    Pxy = {}
    max_posterior = 0.0
    target_class = 'unknown'
    for y in target_attribute_vals:
        print(f"y: {y}")
        p = Py[y]

        # compute product of all the attribute conditional probabilities
        for i, attribute in enumerate([attribute for attribute in attributes if attribute is not list(attributes.keys())[-1]]):
            print(f"X: {attribute}")
            if(attributes[attribute] == 'continuous'):
                P_x_given_y = gaussian(X[i], Pxy_params[attribute][y]['mean'], Pxy_params[attribute][y]['variance']) 
                p *= P_x_given_y
                print(f"mean = {Pxy_params[attribute][y]['mean']}, variance = {Pxy_params[attribute][y]['variance']}")
                print(f"P({attribute}={X[i]}|{y}) = {P_x_given_y}")
            
            elif(attributes[attribute] == 'cardinal'):
                print("Cardinal attributes not yet implemented!")
                return    

        Pxy[y] = p
        if (p > max_posterior):
            max_posterior = p
            target_class = y

        print(f"y: {y}, prior_p: {Py[y]}, posterior_p = {p}")

    print(f"Posterior probabilities: {Pxy}")
    print(f"Target Class: {target_class}") 
    return target_class  

def get_y_vals(instances):
    y = [instance[-1] for instance in instances]
    y_vals, counts = np.unique(y, return_counts=True)
    return y_vals, counts


def gaussian(x, mu, sig):
    sig += 1.e-9; # add a small constant to sigma to avoid division by zero
    g = min(np.exp(-0.5* ((x-mu)/sig)**2) / (np.sqrt(2*np.pi)*sig), 1.0)
    return g


# Sample training data: Each instance has 3 continuous attributes X1, X2, and X3 and the binary target attribute/class Y.
attributes = {'X1' : 'continuous', 'X2' : 'continuous', 'X3' : 'continuous', 'Y' : 'cardinal'}
target_attribute = 'Y'
training_instances = [ [0.8, 0.4, 39.5, 'flu' ],
                       [0.0, 0.8, 37.8, 'cold'],
                       [0.4, 0.4, 37.8, 'flu' ],
                       [0.4, 0.0, 37.8, 'cold'],
                       [0.8, 0.8, 37.8, 'flu' ] ]
target_attribute_vals = get_y_vals(training_instances)[0]

Py, Pxy_params = train_naive_bayes(training_instances, attributes)
print("Naive Baye's model has been trained!")

print(f"Class prior probabilities: {Py}")
print(f"Conditional probability gaussian paramenets: {Pxy_params}")

test_instance = [0.8, 0.4, 37.8]
predict_class(test_instance, Py, Pxy_params, attributes, target_attribute_vals)
