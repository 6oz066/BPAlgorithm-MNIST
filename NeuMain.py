from random import random
import tensorflow as tf
import numpy as np


# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Input layer class
class Input_layer:
    def __init__(self,in_value):
        # input value
        value = in_value


class hidden_layer:
    def __init__(self,i_line,i_col):
        # Number of element per level
        self.num_line = i_line
        # Number of level
        self.num_col = i_col

        # Build the structure
        self.weight_martix = np.random.rand(self.num_col,self.num_line)
        self.threshold_martix = np.random.rand(self.num_col,self.num_line)

        # Output
        self.Output_value=np.zeros((self.num_col,self.num_line))

    # Tool Function - Do not use directly
    def output_function(self,input,function,index):
        return function(input-self.threshold_martix[index[0],index[1]])

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

    def backward_train(self):
        pass


class Output_layer:
    def __init__(self,num_element):
        # Weight
        self.weight = np.random.rand(1,num_element)
        self.threshold = random()

    # Function for outputting calculation
    def output_result(self,input):
        # print(np.sum(np.dot(input,self.weight.T))) # For test usage
        return int(self.output_function(np.sum(np.dot(input,self.weight.T)),sigmoid))

    # Tool Function - Not use directly
    def output_function(self,input,function):
        return function(input-(self.threshold))

    # Loss Function

if __name__ == '__main__':

    # 加载MNIST数据集
    mnist = tf.keras.datasets.mnist

    # 将数据集分为训练集和测试集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 归一化数据
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 查看数据集的形状
    print(x_train.shape, y_train.shape)  # 训练集的形状
    print(x_test.shape, y_test.shape)   # 测试集的形状
    x_train_flat = x_train.reshape(-1, 28 * 28)  # -1 表示自动计算该维度的大小

    # 扁平化测试集图像
    x_test_flat = x_test.reshape(-1, 28 * 28)

    # 检查扁平化后的形状
    print("扁平化后的训练集图像形状:", x_train_flat.shape)
    print("扁平化后的测试集图像形状:", x_test_flat.shape)

    a=hidden_layer(784,2)
    b=Output_layer(784)
    print(b.output_result(a.forward_calculate(x_train_flat[100,:])))
    print(y_train[1])