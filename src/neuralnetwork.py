#!/home/th3rudite/anaconda3/bin/python

#from __future__ import print_function
import torch #tensor 
import torch.nn as nn #neural network

def listString2list(str):
    '''
        Converts sting list to a list
    '''
    outlist = []
    #clear junk from line
    str = str.replace(", ", " ").replace("["," ").replace("]", " ") 
    for val in str.split():
        #print(val)
        if val.isdigit():
            outlist.append(int(val))
            
    return outlist


def readData(filename): 
    '''
        Reads data from Datafile
    '''
    Unsorted = []
    Sorted = []
    dataset = []
    with open(filename, "r") as fdata:
        haltCounter = 0
        for line in fdata:
            haltCounter += 1
            tmp = line.split(";")
            #Unsorted = tmp[0] 
            Unsorted = listString2list(tmp[0])
            #Sorted = tmp[1] 
            Sorted = listString2list(tmp[1])

            # print("unsorted: ")
            # print(Unsorted)
            # print(Unsorted[0])
            # print("Sorted: ")
            # print(Sorted)
            # print("\n")
            intensor = torch.FloatTensor(Unsorted)
            outtensor = torch.FloatTensor(Sorted)
            dataset.append((intensor, outtensor))

            if haltCounter > 50: break
    
    return dataset

    

#reading datafile
dataset = readData("../dataset/data.txt")
print(dataset[0])

dtype = torch.float
device = torch.device("cpu") #running on cpu

#batch size 
batch_size = 1 #usually 32, 64, or 128
#input, output, hidden dimension
input_dim = 10 
output_dim = 10
hidden_dim = 10



dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
# print(x)
# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)


learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
