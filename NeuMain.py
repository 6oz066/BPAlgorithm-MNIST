from random import random
import tensorflow as tf
import numpy as np
import math

"""
Tool Function Design
"""
# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid, use the parameter from f(x)
def d_sigmoid(f):
    # Jump out of vanishing gradient
    if f==1.0 or f==0:
        return 0.001*random()
    else:
        return f*(1.0-f)

"""
Class Function Design
"""

# Input layer class
class Input_layer:
    def __init__(self,in_value):
        # input value
        value = in_value


class hidden_layer:
    def __init__(self,i_line,i_col,learning):
        # Number of element per level
        self.num_line = i_line
        # Number of level
        self.num_col = i_col
        # Learning rate
        self.learning = learning

        # Build the structure
        self.weight_martix = np.random.rand(self.num_col,self.num_line)
        self.threshold_martix = np.random.rand(self.num_col,self.num_line)
        self.bias = np.zeros((self.num_col,self.num_line))
        self.loss = np.zeros((self.num_col,self.num_line))

        # Output
        self.Output_value=np.zeros((self.num_col,self.num_line))

    # Tool Function - Do not use directly
    def output_function(self,input,function,index):
        return function(input-self.threshold_martix[index[0],index[1]])

    # For test data, do not use while training
    def forward_calculate(self,Input_martix):
        for i in range(self.num_col):
            for j in range(self.num_line):
                if(i==0):
                    index = list([i, j])
                    self.Output_value[i,j] = self.output_function(np.sum(np.dot(Input_martix, self.weight_martix[i, :])),sigmoid,index)
                else:
                    index = list([i, j])
                    self.Output_value[i,j] = self.output_function(np.sum(np.dot(self.Output_value[i-1,:], self.weight_martix[i, :])),sigmoid,index)

        return self.Output_value[-1]

    # For train data, which will import forward method automatically
    def backward_train(self,x_train,y_train,Output_layer):
        for i in range(x_train.shape[0]):
            # Run the function to update forward train data
            Output_layer.output_result(self.forward_calculate(x_train[i,:]))

        # Use the loss from output layer to calculate
        for i in reversed(range(x_train.shape[0])):
            if i==0:
                hidden_bias = Output_layer.backward_train(y_train[i,:])

            else:
                pass


    def bias_function(self):
        pass

    # Refresh the loss martix after calculating
    def loss_function(self,bias_martix,weight_martix):
        Ek = 1/2 * np.sum(np.dot(bias_martix,weight_martix))
        return Ek


class Output_layer:
    def __init__(self,num_element,learning):
        # Weight
        self.weight = np.random.rand(1,num_element)
        self.threshold = random()
        self.output = 0
        self.learning = learning
        self.bias = 0

    # Function for outputting calculation
    def output_result(self,input):
        # print(np.sum(np.dot(input,self.weight.T))) # For test usage
        self.output = self.output_function(np.sum(np.dot(input,self.weight.T)),sigmoid)
        return self.output

    # Tool Function - Not use directly
    def output_function(self,input,function):
        return function(input-(self.threshold))

    def backward_learning(self,y_train):
        self.bias = self.bias_function(y_train,d_sigmoid)
        self.weight -= self.learning* self.bias
        return self.bias

    def bias_function(self,y_train,dfunction):
        loss = self.loss_function(y_train)
        d_f = dfunction(self.output)
        return loss*d_f

    # Loss Function
    def loss_function(self,y_train):
        E_k=1/2*(self.output-y_train)**2
        return E_k

class NeuralNetwork:


"""
Main Function
"""

if __name__ == '__main__':

    # Load MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Standardize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_std, y_test_std = y_train / 255.0, y_test / 255.0

    # The shape of the array
    print(x_train.shape, y_train.shape)  # 训练集的形状
    print(x_test.shape, y_test.shape)   # 测试集的形状

    # Flat into 1-dimension array
    x_train_flat = x_train.reshape(-1, 28 * 28)  # -1 表示自动计算该维度的大小
    x_test_flat = x_test.reshape(-1, 28 * 28)

    # Shape of flat
    print("扁平化后的训练集图像形状:", x_train_flat.shape)
    print("扁平化后的测试集图像形状:", x_test_flat.shape)

    # Learning rate
    Learning = 0.5

    # For test usage
    a=hidden_layer(784,2,Learning)
    b=Output_layer(784,Learning)
    print(b.output_result(a.forward_calculate(x_train_flat[100,:])))
    print(y_train[1])
    print(b.weight)
    print(b.output)
    b.backward_learning(y_train_std[1])
    print(b.weight)