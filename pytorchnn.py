import torch
import torch.nn as nn
import torch.nn.functional as F

n_input, n_hidden, n_output = 10, 3, 1

## initialize tensor for inputs, and outputs 
x = torch.randn((1, n_input))
y = torch.randn((1, n_output))

## initialize tensor variables for weights 
w1 = torch.randn(n_input, n_hidden) # weight for hidden layer
w2 = torch.randn(n_hidden, n_output) # weight for output layer


Tensor_Input = []
Tensor_Output = []
with open('dataset/data.txt', 'r') as file:
    for line in file:
        replaced = line.replace('[', '').replace(',', '').replace(' ', '').replace(']','').replace('\n', '')
        split = replaced.split(';')
        inputline = split[0]
        outputline = split[1]

        temp_input_array = []
        for char in inputline:
            temp_input_array.append(int(char))
        Tensor_Input.append(temp_input_array)

        temp_output_array = []
        for char in outputline:
            temp_output_array.append(int(char))
        Tensor_Output.append(temp_output_array)

input = torch.FloatTensor(Tensor_Input)
output = torch.FloatTensor(Tensor_Output)

class NN(nn.Module):
    pass


print(output)
            
# print(Tensor_Input)