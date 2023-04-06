'''
    Implementation of a Naive Bayes Classifier. The target attribute/class label is assumed to be of cardinal type by default.
'''

import numpy as np
import sys

# to train the model, we need to compute all the prior probabilities of each class and the conditional probabilities of each attribute given each class.
def train_naive_bayes(instances, attributes, smoothing = 'laplace'):

    # smoothing parameters
    MIN_VARIANCE = 1.e-9
    EPS = 1.e-9
    LAPLACE_ALPHA = 1

    print(f"Smoothing option: {smoothing}")

    # get attribute values    
    x_vals = get_cardinal_attribute_vals(instances, attributes)
    y_vals, counts = get_y_vals(instances)
    
    print(f"Attribute values: {x_vals}")

    # compute class prior probabilities
    Py = {}
    N = len(instances)
    for i, y in enumerate(y_vals):
        Py[y] = counts[i]/N
    #print(f"Prior class probabilities: {Py}")

    Pxy = {}
    for i, attribute in enumerate([attribute for attribute in attributes if attribute is not list(attributes.keys())[-1]]):
        print(f"\nAttribute: {attribute}")
        # compute gaussian distribution parameters for attribute conditional probabilities
        if(attributes[attribute] == 'continuous'):
            Pxy[attribute] = {}
            for y in y_vals:
                Pxy[attribute][y] = {}
                attribute_vals = [instance[i] for instance in instances if instance[-1] == y]
                print(f"Attribute values for given class: {attribute_vals}")
                 # compute sample mean and variance of attribute values of instances belonging to this class
                Pxy[attribute][y]['mean'] = sum(attribute_vals)/len(attribute_vals)
                Pxy[attribute][y]['variance'] = max(MIN_VARIANCE, np.sqrt(sum([(val - Pxy[attribute][y]['mean'])**2 for val in attribute_vals])/len(attribute_vals)))
                
            print(f"Attribute gaussian distribution parameters: {Pxy[attribute]}")                

        elif(attributes[attribute] == 'cardinal'):
            Pxy[attribute] = {}
            for y in y_vals:
                print(f"y: {y}")
                Pxy[attribute][y] = {}
                all_attribute_vals = [instance[i] for instance in instances if instance[-1] == y]
                print(f"Attribute values of instances belonging this class: {all_attribute_vals}")
                
                # count relative frequencies for each attribute value given the class
                x_vals_unique, x_val_counts = np.unique(all_attribute_vals, return_counts=True)
                x_vals_unique_counts = {x_val:count for x_val, count in zip(x_vals_unique, x_val_counts)}
                print(f"Unique val counts: {x_vals_unique_counts}")
                count_y = len(all_attribute_vals)
                
                M = len(x_vals[attribute])
                for x_val in x_vals[attribute]: 
                    if(x_val in x_vals_unique_counts):
                        count_x_y = x_vals_unique_counts[x_val]    
                    else:
                        count_x_y = 0

                    if(smoothing == None):
                        Pxy[attribute][y][x_val] = count_x_y/count_y
                    elif(smoothing == 'epsilon'):
                        Pxy[attribute][y][x_val] = max(EPS, count_x_y/count_y)
                    elif(smoothing == 'laplace'):
                        Pxy[attribute][y][x_val] = (count_x_y + LAPLACE_ALPHA)/(count_y + LAPLACE_ALPHA*M)

                print(f"Attribute value relative frequencies: {Pxy[attribute][y]}")            

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
            
            elif(attributes[attribute] == 'cardinal'):
                
                P_x_given_y =  Pxy_params[attribute][y][X[i]]          
                p *= P_x_given_y
                
            print(f"P({attribute}={X[i]}|{y}) = {P_x_given_y}") 

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


def get_cardinal_attribute_vals(instances, attributes):
    
    attribute_values = {}
    for i, attribute in enumerate([attribute for attribute in attributes if attribute is not list(attributes.keys())[-1]]):
        if(attributes[attribute] == 'cardinal'):
            x = [instance[i] for instance in instances]
            x_vals = np.unique(x, return_counts=False)
            attribute_values[attribute] = list(x_vals)
        
    return attribute_values


def gaussian(x, mu, sig):
    g = min(np.exp(-0.5* ((x-mu)/sig)**2) / (np.sqrt(2*np.pi)*sig), 1.0)
    return g

'''
# Sample training data: Each instance has 3 continuous attributes X1, X2, and X3 and the binary target attribute/class Y.
attributes = {'X1' : 'continuous', 'X2' : 'continuous', 'X3' : 'continuous', 'Y' : 'cardinal'}
training_instances = [ [0.8, 0.4, 39.5, 'flu' ],
                       [0.0, 0.8, 37.8, 'cold'],
                       [0.4, 0.4, 37.8, 'flu' ],
                       [0.4, 0.0, 37.8, 'cold'],
                       [0.8, 0.8, 37.8, 'flu' ] ]
'''

# Sample training data: Each instance has 3 attributes home_owner, marital_status, and annual_income and the binary target attribute/class defaulted_borrower.
attributes = {'home_owner' : 'cardinal', 'marital_status' : 'cardinal', 'annual_income' : 'continuous', 'defaulted_borrower' : 'cardinal'}
training_instances = [ ['yes', 'single',   125, 'no' ],
                       ['no', 'married',   100, 'no' ],
                       ['no', 'single',    70,  'no' ],
                       ['yes', 'married',  120, 'no' ],
                       ['no', 'divorced',  95,  'yes'], 
                       ['no', 'married',   60,  'no' ], 
                       ['yes', 'divorced', 220, 'no' ], 
                       ['no', 'single',    85,  'yes'], 
                       ['no', 'married',   75,  'no' ], 
                       ['no', 'single',    90,  'yes'] ]



target_attribute_vals = get_y_vals(training_instances)[0]

Py, Pxy_params = train_naive_bayes(training_instances, attributes, 'laplace')
print("Naive Baye's model has been trained!")

print(f"Class prior probabilities: {Py}")
print(f"Conditional probability paramenets: {Pxy_params}")

#test_instance = [0.8, 0.4, 37.8]
test_instance = ['no', 'married', 120]
predict_class(test_instance, Py, Pxy_params, attributes, target_attribute_vals)
