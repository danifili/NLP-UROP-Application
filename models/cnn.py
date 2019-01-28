import torch
from torch import nn
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_size, output_size, kernel_size)
        self.tanh = nn.Tanh()
    
    def forward(self, feature_vectors, max_pool=True):
        conv_out = self.conv(feature_vectors.permute(0, 2, 1).float())  
        post_activation = self.tanh(conv_out)
        if max_pool:
            return post_activation.max(2)[0]
        else:
            return post_activation.mean(2)
    

if __name__ == "__main__":
    cnn = CNN(100, 43, 3)
    print cnn(Variable(torch.randn(5, 10, 100)), False)

