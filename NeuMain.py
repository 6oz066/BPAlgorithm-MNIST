from random import random
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


"""
Tool Function Design
"""
# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid, use the parameter from f(x)
def d_sigmoid(f):
    # Jump out of vanishing gradient
    if f==1.0:
        return 0.001*random()*f
    else:
        return f*(1.0-f)

def d_ad_sigmoid(f):
    return f*(1-f)


"""
Class Function Design
"""

# Input layer class
class Input_layer:
    def __init__(self,in_value):
        # input value
        self.value = in_value

    def active(self,i):
        return self.value[i]

    def topreditct(self,x_test):
        self.value = x_test


class Hidden_layer:
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

        # Run the function to update forward train data
        Output_layer.output_result(self.forward_calculate(x_train))

        # Use the loss from output layer to calculate
        for i in reversed(range(self.num_col)):
            if i==self.num_col-1:
                hidden_bias = Output_layer.backward_learning(y_train)
                for j in range(self.num_col):
                    self.bias[i,j] = hidden_bias*self.weight_martix[i,j]*d_sigmoid(self.Output_value[i,j])
                    self.weight_martix[i,j] -= self.bias[i,j] * self.learning
                    self.threshold_martix[i,j] -= self.bias[i,j] *self.learning
            else:
                self.bias[i,:] = self.bias_function(i,d_ad_sigmoid)
                self.weight_martix[i,:] -= self.bias[i,:] * self.learning
                self.threshold_martix[i,:] -= self.bias[i,:] * self.learning


    # Calculate bias except the last level
    def bias_function(self,i,function):
        return np.dot(function(self.Output_value[i,:]),self.loss_function(i))

    # Calculate loss except the last level
    def loss_function(self,i):
        return np.dot(self.bias[i-1,:],self.weight_martix[i,:])


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
        self.threshold -= self.learning* self.bias

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
    def __init__(self, learning_rate, hiddenlevel, x_train_m, y_train_m, x_test_m, y_test_m):
        self.input_layer = Input_layer(x_train_m)
        self.learning_rate = learning_rate
        self.hiddenlevel= hiddenlevel
        self.hidden_layer = Hidden_layer(x_train_m.shape[1], self.hiddenlevel, self.learning_rate)
        self.output_layer = Output_layer(x_train_m.shape[1], self.learning_rate)
        self.x_train = x_train_m
        self.y_train = y_train_m
        self.x_test = x_test_m
        self.y_test = y_test_m

        self.correct=0

    def train(self):
        print("Training active")
        for i in tqdm(range(len(self.x_train)),desc="Training process"):
            temp_output=self.output_layer.output_result(self.hidden_layer.forward_calculate(self.input_layer.active(i)))
            self.hidden_layer.backward_train(self.x_train[i],self.y_train[i],self.output_layer)

        print("Training finished for loop times:",len(self.x_train))


    def predict(self):
        self.input_layer.topreditct(self.x_test)
        self.correct=0
        dynamic_correct_rate =0
        dynamic_correct_point = np.zeros(len(self.x_test),dtype=float)
        for i in tqdm(range(len(self.x_test)),desc="Testing process"):
            predict_result = self.output_layer.output_result(self.hidden_layer.forward_calculate(self.input_layer.active(i)))
            if predict_result == self.y_test[i]:
                self.correct+=1
                dynamic_correct_rate = self.correct/(i+1)
                dynamic_correct_point[i]=(dynamic_correct_rate)

        plot_x = np.linspace(0,len(self.x_test),len(self.x_test))
        plot_y = dynamic_correct_point
        plt.plot(plot_x,plot_y)
        plt.legend(["Loop time","Correct Rate"])
        plt.title("Dynamic Predicted Correct Rate")
        plt.show()

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

    model = NeuralNetwork(0.1,3,x_train_flat,y_train_std,x_test_flat,y_test)
    print(model.hidden_layer.num_col)
    # model.train()
    model.predict()

    # For test usage
    # Learning rate
    #     Learning = 0.5
    # a=Hidden_layer(784,2,Learning)
    # b=Output_layer(784,Learning)
    # print(b.output_result(a.forward_calculate(x_train_flat[100,:])))
    # print(y_train[1])
    # print(a.weight_martix)
    # for i,j in zip(x_train_flat,y_train_std):
    #     a.backward_train(i,j,b)
    # print(a.weight_martix)