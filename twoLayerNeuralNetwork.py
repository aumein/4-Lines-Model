import Data as d
import numpy as np

#/-----------------------------------------------------------------/#
# Currently we have 2 layer in the neural network and each layer has 
# v nodes. General formula should be z = W * A + B, where 
# g(z) = next layer and g() is the activation function. Input layer 
# A0 will be defined in the main function via the Data class. gl o7
#/-----------------------------------------------------------------/#

alpha = 0.03
epsilon = 1e-15

def activation(z):
    #ReLU
    tot = np.sum(z)
    return (np.maximum(0,z)/tot)

def deriv_activation(z):
    #From ReLU trick
    return z > 0

def softmax(x):
    x=x.astype(float)
    if x.ndim==1:
        S=np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim==2:
        result=np.zeros_like(x)
        M,N=x.shape
        for n in range(N):
            S=np.sum(np.exp(x[:,n]))
            result[:,n]=np.exp(x[:,n])/S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")

# Forward Propagation function, which will make predictions 
# using W1, W2, b1 and b2. Input layer is A0.
def forward_prop(W1, W2, b1, b2, A0, n):
    
    Z1 = W1.dot(A0.T) + b1
    A1 = activation(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

# TODO: find new formulas for adjusting weights and biases
# Note that labels is (2xn) representing the correct label for all n diagrams
# and predictions is (2xn) representing the guess the model makes
def backward_prop(Z1, A1, W2, predictions, labels, A0, n, Z2):
    
    dz2 = -(predictions-labels)                 #dz2 is 2xn
    dw2 = 1/n * dz2.dot(A1.T)                   #dz2 is 2xn, A1.T is nxv, dw2 is 2xv
    db2 = 1/n * np.sum(dz2,1, keepdims=True)                  #db2 is 2x1
    
    dz1 = W2.T.dot(dz2) * deriv_activation(Z1)                   #double check this line, but dz1 is vxn
    dw1 = 1/n * dz1.dot(A0)                                      #dw1 should be vx400
    db1 = 1/n * np.sum(dz1,1, keepdims=True)       #double check, but should be vx4
    
    return dw1, db1, dw2, db2

def prediction(A2):
    return np.argmax(A2, 0)


#Function to update parameters, new set of training can continue after this
#SHOULD BE GOOD AS IS
def update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2

    # Set the maximum allowed norm for the gradient
    max_gradient_norm = 1.0  # Adjust as needed

    # Calculate the norm of the gradient
    b2_norm = np.linalg.norm(b2)
    b1_norm = np.linalg.norm(b1)
    W1_norm = np.linalg.norm(W1)
    W2_norm = np.linalg.norm(W2)

     #Clip the gradient if its norm exceeds the specified threshold
    if b2_norm > max_gradient_norm:
        clip_factor = max_gradient_norm / (b2_norm + 1e-6)  # small constant to avoid division by zero
        b2 = b2 * clip_factor
    else:
       b2 = b2

    if b1_norm > max_gradient_norm:
        clip_factor = max_gradient_norm / (b1_norm + 1e-6)  # small constant to avoid division by zero
        b1 = b1 * clip_factor
    else:
        b1 = b1

    if W1_norm > max_gradient_norm:
        clip_factor = max_gradient_norm / (W1_norm + 1e-6)  # small constant to avoid division by zero
        W1 = W1 * clip_factor
    else:
        W1 = W1
        
    if W2_norm > max_gradient_norm:
        clip_factor = max_gradient_norm / (W2_norm + 1e-6)  # small constant to avoid division by zero
        W2 = W2 * clip_factor
    else:
        W2 = W2
    
    return W1, W2, b1, b2


def init_params(n, v):
    W1 = np.random.rand(v, 1600)
    W2 = np.random.rand(4,v)
    B1 = np.random.rand(v, n)
    B2 = np.random.rand(4,n)
    
    return W1, W2, B1, B2

def accuracy(predictions, Y, n):
    count = 0
    for i in range(n):
        if predictions[i] == Y[i]:
            count += 1
            
    return count / n

# -------------------------------- #
def task2(n):
    v = 6  #nodes per hidden layer
    
    training = d.Data(n)
    a0, Y = training.generateDataset2(n)
    Y = Y.T
    
    W1, W2, b1, b2 = init_params(n, v)
    
    for i in range(10000):
        
        Z1, A1, Z2, A2 = forward_prop(W1, W2, b1, b2, a0, n)            #Note that A2 contains the prob for: row0 = P(dangerous) and row1 = P(not dangerous)                              #predictions for all n training sets
        dw1, db1, dw2, db2 = backward_prop(Z1, A1, W2, A2, Y, a0, n, Z2)
        W1, W2, b1, b2 = update_parameters(W1, W2, b1, b2, dw1, dw2, db1, db2, alpha)
        
        if i%1000 == 0: 
            acc = accuracy(prediction(A2).flatten(), Y.flatten(), n)
            loss = -np.sum(Y * np.log(A2 + epsilon)) / n
            print(f"Current Generation: {i}, Cost: {loss}")
            print("Accurracy: ", acc)  
            
            a0, labels = training.generateDataset(n)
            labels = labels.T
            Y = labels[0,:]   
            
    
    print("Training complete, starting test set")
    total = 0
    
    a0, labels = training.generateDataset(n)
    labels = labels.T
    Y = labels[0,:] 
        
    for i in range(1):
        
        _, _, _, A2 = forward_prop(W1, W2, b1, b2, a0, n)
        total += accuracy(prediction(A2).flatten(), Y.flatten(), n)
        
        if i%1 == 0: 
            print("Number of Diagrams tested: ", (i+1)*n)
            print("Accurracy: ", total/(i+1))
            #a0, labels = training.generateDataset(n)
            #labels = labels.T
            #Y = labels[0,:] 
# ----------------------------------------------------------------- #


def main():
    task2(1000)
            
if __name__  == '__main__':
    main()