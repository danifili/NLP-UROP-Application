from torch import nn
import torch
from torch.autograd import Variable

class GRU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, output_size, num_layers = 1, batch_first=True)

    def forward(self, feature_vector):
        output, hidden = self.gru(feature_vector)
        return output.mean(1)

if __name__ == "__main__":
    batch_size = 16
    input_size = 200
    output_size = 600
    sentence_length = 100
    kernel_size = 3

    encoder = GRU(input_size, output_size)
    feature_vectors = Variable(torch.randn(batch_size, sentence_length, input_size))
    output = encoder(feature_vectors)
    print output.data.shape
    assert output.data.shape == (batch_size, output_size)