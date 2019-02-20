#!/home/erud1t3/anaconda3/bin/python

import torch #for torch tensors
import torch.nn as nn #for torch neural networks


Tensor_Input = []
Tensor_Output = []
N_EPOCHS = 100000
DATASET_SIZE = 10

with open('dataset/data.txt', 'r') as file:
    counter = 0
    for line in file:
        
        counter += 1    
        if counter > DATASET_SIZE: break # determines how much data to train upon

        # lines below process the data
        replaced = line.replace('[', '').replace(',', '').replace(' ', '').replace(']','').replace('\n', '')
        split = replaced.split(';')
        inputline = split[0]
        outputline = split[1]

        temp_input_array = []
        for char in inputline: temp_input_array.append(int(char))
        Tensor_Input.append(temp_input_array)

        temp_output_array = []
        for char in outputline: temp_output_array.append(int(char))
        Tensor_Output.append(temp_output_array)



class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()

        # parameters
        self.inputSize = 10 # 1000 for list of ten unsorted digits
        self.outputSize = 10 # 10 for a list of ten sorted digits
        self.hiddenSize = 10 # 10 for hidden layer of ten nodes
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 1000 X 10 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 10 X 10 tensor
        # print('W1 : ' + str(self.W1.size()))
        # print('W2 : ' + str(self.W2.size()))
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 1000 X 10 matrix product
        self.z2 = self.sigmoid(self.z) # sigmoid activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1 - s)  # derivative of sigmoid
    
    def backward(self, X, y, o):
        '''
            Back propagation
            src: https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
        '''
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        '''
            forward + backward pass for training
        '''
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        torch.save(model, "NN") # we will use the PyTorch internal storage functions
        # torch.load("NN") # you can reload model with all the weights and so forth with:
        
    def predict(self):
        '''
            Predict data based on trained weights
        '''
        print ("\nPredicted data based on trained weights: \n")
        print ("Input : \n" + str(xPredicted_unscaled))
        print ("Target: \n" + str(target))
        print ("Output: \n" + str(torch.round(9 * self.forward(xPredicted))))


X = torch.FloatTensor(Tensor_Input) # 1000 x 10 
y = torch.FloatTensor(Tensor_Output) # 1000 x 10

print(X.size())
print(y.size())

xPredicted_unscaled = xPredicted = X[2]
target = y[2]

X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted_unscaled, 0)

X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted_unscaled, xPredicted_max)
y = y / 9  # max test score is 100

# print(y[0])

NN = Neural_Network()
for i in range(N_EPOCHS):  # trains the NN 1,000 times
    print ("Epoch " + str(i) + " | Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()
