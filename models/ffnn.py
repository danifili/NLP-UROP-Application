import torch
from torch import nn
from torch.autograd import Variable

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize feed forward neural net classifier
        :param input_size: size of the feature vectors
        :param hidden_sizes: non-empty list of sizes of the hidden layers of this arquitecture.
        :param output_size: number of classes to be classified
        """
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_to_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature_vectors):
        """
        Output the probabilities of input in a certain class
        :param feature_vectors: Variable wrapping torch tensor of dimensions batch_size x input_size
        :return: the probabilities, Variable wrapping torch tensor of dimensions batch_size x output_size
        """
        hidden_layer = self.input_to_hidden(feature_vectors)
        post_activation = self.relu(hidden_layer)
        pre_output = self.hidden_to_out(post_activation)
        output = self.sigmoid(pre_output)
        return output

    
if __name__ == "__main__":
    inp_size, hid_size, out_size = 100, 23, 10
    ffnn = FFNN(inp_size, hid_size, out_size)
    batch_size = 7
    vec = Variable(torch.randn(batch_size, inp_size))
    print (ffnn(vec))

        
