# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
# import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import torch.nn.functional as F
from torch.distributions.beta import Beta
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

"""
    BiLSTM + CNN
"""

char_label = [torch.LongTensor([1, 0, 2, 8, 6, 13, 11, 16, 10, 3]), torch.LongTensor([2, 11, 9, 12, 0, 13, 4, 14]),
              torch.LongTensor([4, 18, 15, 4, 10, 14, 7, 11, 10]),  torch.LongTensor([5, 16, 15, 16, 13, 4]),
              torch.LongTensor([9, 11, 15, 7, 17, 0, 15, 7, 11, 10]), torch.LongTensor([16, 14, 4, 14])]

class LSTMCNN(nn.Module):

    def __init__(self, vocab, hidden_size, num_layers, batch, bidirectional, filter_sizes, out_channels,
                 size_average=True, alpha=0.25, num_classes=6, gamma=2):
        super(LSTMCNN, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        vocab_size = vocab.vectors.size()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])  # Embedding.weight.requires_grad default: True
        self.embedding.weight.data.copy_(vocab.vectors)
        # self.char_embedding = nn.Embedding(19, 300)
        # self.char_linear = nn.Linear(300, 6)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=vocab_size[1], hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(i, 500)) for i in filter_sizes])
        # self.CNN_dropput = F.dropout(0.5, training=self.training, inplace=True)
        self.CNN_up = nn.Linear(len(filter_sizes) * out_channels, 2 * len(filter_sizes) * out_channels)
        self.CNN_fc = nn.Linear(2 * len(filter_sizes) * out_channels, out_channels // 2)
        self.relu = nn.ReLU()
        # self.CNN_fc1 = nn.Linear(out_channels // 2, out_channels // 4)
        self.tanh = nn.Tanh()
        self.CNN_out = nn.Linear(out_channels // 2, 6)
        self.up_dim = nn.Linear(200, 500)
        # self.ac_tanh = nn.Tanh()
        # self.linear = nn.Linear(hidden_size * 2, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, 6)
        self.combine_linear = nn.Linear(512, 6)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    # def init_hidden(self, bidirectional):
    #     """
    #         h_0 of shape (num_layers * num_directions, batch, hidden_size)
    #         c_0 of shape (num_layers * num_directions, batch, hidden_size)
    #     """
    #     # return (torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True),
    #     #         torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True)) \
    #     #     if bidirectional else (torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True),
    #     #                            torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True))
    #     return (torch.randn((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True),
    #             torch.randn((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True)) \
    #         if bidirectional else (torch.randn((self.num_layers, self.batch, self.hidden_size), requires_grad=True),
    #                                torch.randn((self.num_layers, self.batch, self.hidden_size), requires_grad=True))

    def forward(self, word_id=None, sen_len=None, t_class_id=None, **kwargs):
        # print(word_id[0])
        word_vector = self.embedding(word_id)  # [batch_size, seq_len, embedding_size]
        word_vector = F.dropout(word_vector, p=0.1, training=self.training, inplace=True)
        word_vector_pack = rnn_utils.pack_padded_sequence(word_vector, lengths=sen_len, batch_first=True,
                                                          enforce_sorted=False)
        output, (hn, cn) = self.lstm(word_vector_pack)
        output, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        # print(output.shape)
        out_len_f = (out_len - 1).view(out_len.shape[0], 1, -1)
        out_len_f = out_len_f.repeat(1, 1, self.hidden_size * 2)
        out_len_f = out_len_f.to(device)
        rnn_out_f = torch.gather(output, 1, out_len_f)
        rnn_out_f = torch.squeeze(rnn_out_f, dim=1)
        # rnn_out_f = self.linear(rnn_out_f)
        # rnn_out_f = self.ac_tanh(rnn_out_f)
        # # print(rnn_out_f.shape)
        # rnn_out_f = self.linear2(rnn_out_f)
        output = self.up_dim(output)
        l2c_matrix = output.unsqueeze(1)
        l2c_out = torch.cat([self.conv_and_pool(l2c_matrix, conv) for conv in self.convs], dim=1)
        l2c_out = F.dropout(l2c_out, p=0.5, training=self.training, inplace=True)
        l2c_out = self.CNN_up(l2c_out)
        l2c_out = self.relu(l2c_out)
        l2c_out = self.CNN_fc(l2c_out)
        l2c_out = self.tanh(l2c_out)
        l2c_out = self.CNN_out(l2c_out)
        return l2c_out

    # def mixup_criterion(criterion, pred, y_a, y_b, lam):
    #     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

"""recurrent convolutional neural networks for text classification"""
class RCNN(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, bidirectional):
        super(RCNN, self).__init__()
        self.num_layers = num_layers
        vocab_size = vocab.vectors.size()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])  # Embedding.weight.requires_grad default: True
        self.embedding.weight.data.copy_(vocab.vectors)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=vocab_size[1], hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(0.3)
        self.W = nn.Linear(vocab_size[1] + 2 * hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, word_id=None, sen_len=None, **kwargs):
        word_vector = self.embedding(word_id)  # [batch_size, seq_len, embedding_size]
        word_vector_pack = rnn_utils.pack_padded_sequence(word_vector, lengths=sen_len, batch_first=True,
                                                          enforce_sorted=False)
        output, (hn, cn) = self.lstm(word_vector_pack)
        output, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        max_len = max(sen_len)
        select_idx = torch.arange(0, max_len).to(device)
        word_vector = torch.index_select(word_vector, dim=1, index=select_idx)
        # print(output[0])
        # print(word_vector[0])
        # print(output.shape)
        # print(word_vector.shape)
        # exit()
        output = torch.cat([output, word_vector], 2)
        output = self.tanh(self.W(output)).transpose(1, 2)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        output = self.dropout(output)
        output = self.fc(output)
        # print(output.shape)
        # out_len_f = (out_len - 1).view(out_len.shape[0], 1, -1)
        # out_len_f = out_len_f.repeat(1, 1, self.hidden_size * 2)
        # out_len_f = out_len_f.to(device)
        # rnn_out_f = torch.gather(output, 1, out_len_f)
        # rnn_out_f = torch.squeeze(rnn_out_f, dim=1)
        # # rnn_out_f = self.linear(rnn_out_f)
        # # rnn_out_f = self.ac_tanh(rnn_out_f)
        # # # print(rnn_out_f.shape)
        # # rnn_out_f = self.linear2(rnn_out_f)
        # output = self.up_dim(output)
        # l2c_matrix = output.unsqueeze(1)
        # l2c_out = torch.cat([self.conv_and_pool(l2c_matrix, conv) for conv in self.convs], dim=1)
        # l2c_out = F.dropout(l2c_out, p=0.5, training=self.training, inplace=True)
        # l2c_out = self.CNN_up(l2c_out)
        # l2c_out = self.relu(l2c_out)
        # l2c_out = self.CNN_fc(l2c_out)
        # l2c_out = self.tanh(l2c_out)
        # l2c_out = self.CNN_out(l2c_out)
        return output
