# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.distributions.beta import Beta
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
import numpy as np
"""
    BiLSTM + Attention
    Attention-Based Bidirectional Long Short-Term Memory Networks forRelation Classification
"""


# class LSTMAttention(nn.Module):
#     def __init__(self, vocab, hidden_size, num_layers, batch, bidirectional):
#         super(LSTMAttention, self).__init__()
#         self.num_layers = num_layers
#         self.batch = batch
#         self.vocab_size = vocab.vectors.size()
#         # Embedding.weight.requires_grad default: True
#         self.embedding = nn.Embedding(self.vocab_size[0], self.vocab_size[1])
#         self.embedding.weight.data.copy_(vocab.vectors)
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.lstm = nn.LSTM(input_size=self.vocab_size[1], hidden_size=hidden_size, num_layers=self.num_layers,
#                             batch_first=True, bidirectional=self.bidirectional)
#         self.rac_tanh = nn.ReLU()
#         self.lac_tanh = nn.ReLU()
#         self.rdropout = nn.Dropout(0.5)
#         self.lfropout = nn.Dropout(0.5)
#         self.rlinear1 = nn.Linear(hidden_size * 2, hidden_size)
#         self.llinear1 = nn.Linear(hidden_size * 2, hidden_size)
#         self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 6)
#         # self.word_vec = nn.Linear(self.vocab_size[1], 6)
#         self.rH2M = nn.ReLU()
#         self.rw = nn.Parameter(torch.randn(self.hidden_size * 2), requires_grad=True)
#         self.lH2M = nn.ReLU()
#         self.lw = nn.Parameter(torch.randn(self.hidden_size * 2), requires_grad=True)
#
#     # def init_hidden(self, bidirectional):
#     #     """
#     #         h_0 of shape (num_layers * num_directions, batch, hidden_size)
#     #         c_0 of shape (num_layers * num_directions, batch, hidden_size)
#     #     """
#     #     return (torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True),
#     #             torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True)) \
#     #         if bidirectional else (torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True),
#     #                                torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True))
#
#     def forward(self, word_id=None, sen_len=None, **kwargs):
#         # [batch_size, seq_len, embedding_size]
#         np.random.seed(0)
#         index = 0
#         # lamb = Beta(0.5, 0.5).sample()
#         lamb = np.random.beta(0.5, 0.5)
#         # if not self.training:
#         #     reverse_word_id = word_id
#         #     reverse_sen_len = sen_len
#         word_vector = self.embedding(word_id)
#         # reverse_word_vector = self.embedding(reverse_word_id)
#         # h0, c0 = self.init_hidden(self.bidirectional)
#         # h0 = h0.to(device)
#         # c0 = c0.to(device)
#         word_vector_pack = rnn_utils.pack_padded_sequence(word_vector, lengths=sen_len, batch_first=True,
#                                                           enforce_sorted=False)
#         # reverse_word_pack = rnn_utils.pack_padded_sequence(reverse_word_vector, lengths=reverse_sen_len,
#         #                                                    batch_first=True, enforce_sorted=False)
#         routput, (rhn, rcn) = self.lstm(word_vector_pack)
#         # loutput, (lhn, lcn) = self.lstm(reverse_word_pack)
#         rH, rout_len = rnn_utils.pad_packed_sequence(routput, batch_first=True)
#         # lH, lout_len = rnn_utils.pad_packed_sequence(loutput, batch_first=True)
#         # out_len_f = (out_len - 1).view(out_len.shape[0], 1, -1)
#         # out_len_f = out_len_f.repeat(1, 1, self.hidden_size * 2)
#         # out_len_f = out_len_f.to(device)
#         # out_f = torch.gather(output, 1, out_len_f)
#         # out_f = torch.squeeze(out_f, dim=1)
#         # out_f = self.linear(out_f)
#         # out_f = self.ac_tanh(out_f)
#         rM = self.rH2M(rH)
#         # lM = self.lH2M(lH)
#         # w_T M(1 * d_w ** d_w * T) = M w (T * d_w ** d_w * 1)
#         ralpha = torch.softmax(torch.matmul(rM, self.rw), dim=0).unsqueeze(-1)
#         # lalpha = torch.softmax(torch.matmul(lM, self.lw), dim=0).unsqueeze(-1)
#         rout = rH * ralpha
#         # lout = lH * lalpha
#         rout = torch.sum(rout, 1)
#         # lout = torch.sum(lout, 1)
#         # if self.training:   # Mixup
#         #     index = torch.randperm(word_id.shape[0]).to(device)
#         #     rand_lout = lout[index, :]
#         #     lout = lamb * lout + (1 - lamb) * rand_lout
#         rout = self.rac_tanh(rout)
#         rout = self.rdropout(rout)
#         # rout = self.fc1(rout)
#         # rout = self.fc2(rout)
#         rout = self.rlinear1(rout)
#         # rout = self.rlinear2(rout)
#         # lout = torch.sum(lout, 1)
#         # lout = self.lac_tanh(lout)
#         # lout = self.lfropout(lout)
#         # lout = self.llinear1(lout)
#         # lout = self.llinear2(lout)
#         # index = torch.randperm(word_id.shape[0]).to(device)
#         # rand_lout = rout[index, :]
#         # mixed_lout = lamb * rout + (1 - lamb) * rand_lout
#         # if self.training:
#         #     # l = 1 - ((kwargs['epoch'] - 1) / self.total_epoch) ** 2
#         #     l = kwargs['l']
#         #     mixed_feature = torch.cat((l * rout, (1 - l) * lout), dim=1) # BNN
#         #     fc1_out = self.fc1(mixed_feature)
#         #     fc2_out = self.fc2(fc1_out)
#         #     return fc2_out, index, lamb, lout
#         mixed_feature = torch.cat((rout, lout), dim=1)
#         fc1_out = self.fc1(mixed_feature)
#         fc2_out = self.fc2(fc1_out)
#         return fc2_out


class LSTMAttention(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, batch, bidirectional):
        super(LSTMAttention, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        self.vocab_size = vocab.vectors.size()
        # Embedding.weight.requires_grad default: True
        self.embedding = nn.Embedding(self.vocab_size[0], self.vocab_size[1])
        self.embedding.weight.data.copy_(vocab.vectors)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.vocab_size[1], hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        self.rH2M = nn.Tanh()
        self.rw = nn.Parameter(torch.zeros(self.hidden_size * 2), requires_grad=True)
        self.rac_tanh = nn.Tanh()
        self.rdropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 6)

    # def init_hidden(self, bidirectional):
    #     """
    #         h_0 of shape (num_layers * num_directions, batch, hidden_size)
    #         c_0 of shape (num_layers * num_directions, batch, hidden_size)
    #     """
    #     return (torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True),
    #             torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True)) \
    #         if bidirectional else (torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True),
    #                                torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True))

    def forward(self, word_id=None, sen_len=None, **kwargs):
        # [batch_size, seq_len, embedding_size]
        word_vector = self.embedding(word_id)
        word_vector_pack = rnn_utils.pack_padded_sequence(word_vector, lengths=sen_len, batch_first=True,
                                                          enforce_sorted=False)
        routput, (rhn, rcn) = self.lstm(word_vector_pack)
        rH, rout_len = rnn_utils.pad_packed_sequence(routput, batch_first=True)
        rM = self.rH2M(rH)
        # w_T M(1 * d_w ** d_w * T) = M w (T * d_w ** d_w * 1)
        ralpha = torch.softmax(torch.matmul(rM, self.rw), dim=0).unsqueeze(-1)
        rout = rH * ralpha
        # lout = lH * lalpha
        rout = torch.sum(rout, 1)
        rout = self.rac_tanh(rout)
        fc1_out = self.fc1(rout)
        fc1_out = self.rdropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out
