import torch
from torch import nn
from torch.autograd import Variable

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_to_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature_vectors):
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

        
