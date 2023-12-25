import numpy as np
import Data as d
import matplotlib.pyplot as plt

def shuffle(X, Y):
    indices = np.random.permutation(len(X))
    
    shuffledData = X[indices]
    shuffledLabels = Y[indices]
    
    return shuffledData, shuffledLabels

def shuffle2(X, Y):
    indices = np.random.permutation(len(X))
    Y = Y.T 
    
    shuffledData = X[indices]
    shuffledLabels = Y[indices]
    
    return shuffledData, shuffledLabels.T

def accuracy(predictions, Y, n):
    count = 0
    for i in range(n):
        if predictions[i] == Y[i]:
            count += 1
            
    return count / n

def prediction(A2):
    pred = []
    for ele in A2:
        for _ in ele:
            if _ > 0.5:
                pred.append(1)
            else:
                pred.append(0)
                
    return pred

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    #ReLU
    return np.maximum(0,z)

def deriv_ReLU(z):
    #From ReLU trick
    return z > 0


def initialize_parameters(input_size):
    # Initialize weights and bias
    W = np.random.randn(1, input_size)
    b = np.zeros((1, 1))
    return W, b

def forward_propagation(X, W, b):
    # Compute preactivation
    Z = np.dot(W, X.T) + b

    # Apply activation function (sigmoid)
    A = sigmoid(Z)
    
    return Z, A

def backward_propagation(X, A, Y):
    # Compute the gradient of the loss with respect to the weights and bias
    m = X.shape[0]
    dz = A - Y
    dw = (1/m) * np.dot(dz, X)
    db = (1/m) * np.sum(dz)
    
    return dw, db

def update_parameters(W, b, dw, db, learning_rate):
    # Update weights and bias using gradient descent
    W = W - learning_rate * dw
    b = b - learning_rate * db
    
    return W, b

def train(X, Y, epochs, learning_rate, training, n):
    input_size = X.shape[1]
    W, b = initialize_parameters(input_size)
    cost_arr = []
    for epoch in range(epochs):
        # Forward propagation
        Z, A = forward_propagation(X, W, b)
        
        # Backward propagation
        dw, db = backward_propagation(X, A, Y)
        
        # Update parameters
        W, b = update_parameters(W, b, dw, db, learning_rate)
        
        if epoch % 10000 == 0:
            # Print the cost (optional)
            cost = np.mean(-Y * np.log(A) - (1 - Y) * np.log(1 - A))
            print(f"Epoch {epoch}, Cost: {cost}")
            cost_arr.append(cost)
            
            #X, Y = shuffle(X, Y)
            X, Y = training.generateDataset(n)
            Y = Y[:,0]

    return W, b, cost_arr

def task1(n):
    np.random.seed(42)
    # X is your input data and Y is the corresponding labels
    training = d.Data(n)
    X, Y = training.generateDataset(n)
    Y = Y[:,0]

    # Train the single-layer neural network
    trained_W, trained_b, costs = train(X, Y, epochs=500000, learning_rate=0.02, training=training, n = n)

    X, Y = training.generateDataset(n)
    Y = Y[:,0]

    # Make predictions
    _, predictions = forward_propagation(X, trained_W, trained_b)

    acc = accuracy(prediction(predictions), Y, n)
    print("Accuracy: ", acc)
    return costs
    
# ================================================================================== #

def initialize_parameters2(input_size):
    # Initialize weights and bias
    W = np.random.randn(4, input_size)
    b = np.zeros((4, 1))
    return W, b

def forward_propagation2(X, W, b):
    # Compute preactivation
    Z = np.dot(W, X.T) + b

    # Apply activation function (sigmoid)
    A = ReLU(Z)
    
    return Z, A

def backward_propagation2(X, A, Y):
    # Compute the gradient of the loss with respect to the weights and bias
    m = X.shape[0]
    dz = A - Y
    dw = (1/m) * np.dot(dz, X)
    db = (1/m) * np.sum(dz)
    
    return dw, db

def train2(X, Y, epochs, learning_rate, training, n):
    pltX = []
    pltY = []
    
    epsilon = 1e-15
    input_size = X.shape[1]
    W, b = initialize_parameters2(input_size)
    
    for epoch in range(epochs):
        # Forward propagation
        Z, A = forward_propagation2(X, W, b) 
        
        # Backward propagation
        dw, db = backward_propagation2(X, A, Y)
        
        # Update parameters
        W, b = update_parameters(W, b, dw, db, learning_rate)
        
        if epoch % 15000 == 0:
            pltX.append(epoch)
            
            # Print the cost (optional)
            cost = -np.sum(Y * np.log(A + epsilon)) / n
            print(f"Epoch {epoch}, Cost: {cost}")
            pltY.append(cost)
            
            X, Y = shuffle2(X, Y)
            #X, Y = training.generateDataset2(n)
            #Y = Y[:,0]
            
    plt.plot(pltX, pltY)
    plt.xlabel('Generations')
    plt.ylabel('Cost')
    plt.title('Cost vs Generations')
    
    plt.show()
    
    return W, b    
    
def task2(n):
    np.random.seed(42)
    
    #X is input data, Y is labels
    training = d.Data(n)
    X, Y = training.generateDataset2(n)
    Y = Y.T

    # Train the single-layer neural network
    trained_W, trained_b = train2(X, Y, epochs=220000, learning_rate=0.02, training=training, n = n)

    X, Y = training.generateDataset2(n)
    Y = Y[:,0]

    # Make predictions
    _, predictions = forward_propagation(X, trained_W, trained_b)

    acc = accuracy(prediction(predictions), Y, n)
    print("Accuracy: ", acc)
    
        
def main():
    steps = np.arange(0, 490001, 10000)
    # results_500 = task1(500)
    # results_1000 = task1(1000)
    # results_2500 = task1(2500)
    results_5000 = task1(5000)

    # loss_500 = plt.scatter(steps, results_500, marker="o")
    # loss_1000 = plt.scatter(steps, results_1000, marker="o")
    # loss_2500 = plt.scatter(steps, results_2500, marker="x")
    # loss_5000 = plt.scatter(steps, results_5000, marker="o")
    plt.xlabel("Generations") 
    plt.ylabel("Cost") 
    # plt.plot(steps, results_500, label = "500 examples") 
    # plt.plot(steps, results_1000, label = "1000 examples") 
    # plt.plot(steps, results_2500, label = "2500 examples") 
    plt.plot(steps, results_5000, label = "5000 examples") 
    plt.show()
    task2(500)
            
if __name__  == '__main__':
    main()