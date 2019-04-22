import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    model = {}
    model['w1'] = w1
    model['w2'] = w2
    model['w3'] = w3
    model['w4'] = w4

    model['b1'] = b1
    model['b2'] = b2
    model['b3'] = b3
    model['b4'] = b4
    #IMPLEMENT HERE
    losses = []
    for e in range(epoch):
        l = 0
        if shuffle:
            x_train, y_train = unison_shuffled_copies(x_train, y_train)
        for i in range(x_train.shape[0]//200):
            X, y = x_train[i*200:(i+1)*200], y_train[i*200:(i+1)*200]
            y = np.reshape(y, (y.shape[0],1))
            loss, model = four_nn(X, y, model, 0.1, False)
            l+=loss
        losses.append(l)
        print(e,l)
        
    w1 = model['w1']
    w2 = model['w2']
    w3 = model['w3']
    w4 = model['w4']

    b1 = model['b1']
    b2 = model['b2']
    b3 = model['b3']
    b4 = model['b4']
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    model = {}
    model['w1'] = w1
    model['w2'] = w2
    model['w3'] = w3
    model['w4'] = w4

    model['b1'] = b1
    model['b2'] = b2
    model['b3'] = b3
    model['b4'] = b4

    classifications = four_nn(x_test, y_test, model, 0, True)

    total_correct = 0
    num_samples, num_classified_correctly = np.zeros((num_classes, 1)), np.zeros((num_classes, 1))
    for n in range(x_test.shape[0]):
        y = y_test[n]
        x = x_test[n][:]
        y_pred = classifications[n]
        num_samples[y]+=1
        if (y_pred == y):
            total_correct += 1
            num_classified_correctly[y] += 1
    avg_classification_rate = total_correct/np.float(len(x_test))
    avg_class_classification_rate = num_classified_correctly/num_samples
    
    return avg_classification_rate, avg_class_classification_rate

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(x_train, y_train, model, lr, test=False):
    w1 = model['w1']
    w2 = model['w2']
    w3 = model['w3']
    w4 = model['w4']

    b1 = model['b1']
    b2 = model['b2']
    b3 = model['b3']
    b4 = model['b4']
    
    z1, acache1 = affine_forward(x_train, w1, b1)
    a1, rcache1 = relu_forward(z1)

    z2, acache2 = affine_forward(a1, w2, b2)
    a2, rcache2 = relu_forward(z2)

    z3, acache3 = affine_forward(a2, w3, b3)
    a3, rcache3 = relu_forward(z3)

    F, acache4 = affine_forward(a3, w4, b4)

    if test:
        classifications = np.argmax(F, axis=1)
        return classifications

    loss, dF = cross_entropy(F, y_train)
    

    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)

    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)

    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)

    dX, dW1, db1 = affine_backward(dZ1, acache1)

    model['w1'] -= (dW1*lr)    
    model['w2'] -= (dW2*lr)
    model['w3'] -= (dW3*lr)
    model['w4'] -= (dW4*lr)

    model['b1'] -= (db1*lr)
    model['b2'] -= (db2*lr)
    model['b3'] -= (db3*lr)
    model['b4'] -= (db4*lr)    

    return loss, model

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    Z = np.dot(A, W) + b
    cache = {}
    cache['A'] = A
    cache['W'] = W
    cache['b'] = b
    return Z, cache

def affine_backward(dZ, cache):
    W = cache['W']
    A = cache['A']
    dA = np.dot(dZ, W.T)
    dW = np.dot(A.T, dZ)
    dB = dZ.sum(axis = 0)
    return dA, dW, dB

def relu_forward(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache['Z'] = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache['Z']
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def cross_entropy(F, y):
    
    loss = 0
    all_classes = np.sum(np.exp(F),axis = 1)
    all_classes = np.reshape(all_classes, (all_classes.shape[0],1))
    
    temp = np.log(all_classes)
    temp = temp.ravel()
    y = y.astype(int)
    y = y.ravel()
    
    sel = F[np.arange(F.shape[0]), y]
    L = sel - temp
    loss = np.sum(L)

    loss/=(-F.shape[0])
    
    dF = np.zeros(F.shape)
    one_hot = np.zeros(F.shape)
    
    one_hot[np.arange(one_hot.shape[0]), y] = 1
    
    dF = one_hot - (np.exp(F)/all_classes)
    dF/=(-F.shape[0])
    return loss, dF