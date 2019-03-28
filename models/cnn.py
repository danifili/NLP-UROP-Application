import torch
from torch import nn
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, max_pool=True):
        """
        Initialize encoder cnn
        :param input_size: number of dimensions of input feature vectors
        :param output_size: number of dimenisons of output feature vector
        :param kernel_size: kernel size of convolution
        :param max_pool: If true, performs max pooling, else it performs mean pooling
        """
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_size, output_size, kernel_size)
        self.tanh = nn.Tanh()
        self.max_pool = max_pool
    
    def forward(self, feature_vectors):
        """
        Output the encoding of a sentence given a batch of sequence of word embeddings
        :param feature_vectors: Variable wrapping torch tensor of dimensions batch_size x sentence_length x input_size
        :return: Variable wrapping a torch tensor of dimensions batch_size x output_size
        """
        conv_out = self.conv(feature_vectors.permute(0, 2, 1).float())  
        post_activation = self.tanh(conv_out)
        if self.max_pool:
            return post_activation.max(2)[0]
        else:
            return post_activation.mean(2)
    

if __name__ == "__main__":
    cnn = CNN(100, 43, 3)
    print (cnn(Variable(torch.randn(5, 10, 100))))

