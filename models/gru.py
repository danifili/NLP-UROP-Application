from torch import nn
import torch
from torch.autograd import Variable
from models.attention import Attention
import numpy as np

class GRU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.encoder = nn.GRU(input_size, output_size/4, num_layers = 1, batch_first=True, dropout=0.1, bidirectional=True)
        self.attention = Attention(output_size/2)
        self.linear_filter = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.post_attention = nn.GRU(output_size, output_size/2, num_layers = 1, batch_first=True, dropout=0.1, bidirectional=True)
        self.final_attention = Attention(output_size)    
    def forward(self, feature_vector):
        encoder_output, _ = self.encoder(feature_vector)
        attention = self.attention(encoder_output)
        concat = torch.cat((encoder_output, attention.unsqueeze(1).expand(-1, encoder_output.data.shape[1], -1)), 2)
        context, _ = self.post_attention(concat)
        final_output = self.final_attention(context)
        return final_output


if __name__ == "__main__":
    batch_size = 16
    input_size = 200
    output_size = 600
    sentence_length = 100
    kernel_size = 3

    encoder = GRU(input_size, output_size)
    feature_vectors = Variable(torch.randn(batch_size, sentence_length, input_size))
    output = encoder(feature_vectors)
    #print (output.data.shape)
    assert output.data.shape == (batch_size, output_size)