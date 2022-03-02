# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, vocab, num_filters, filter_sizes):
        super(CNN, self).__init__()
        vocab_size = vocab.vectors.size()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, vocab_size[1])) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_filters)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(num_filters, 6)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, word_id, **kwargs):
        out = self.embedding(word_id)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(self.tanh(out))
        out = self.fc1(out)
        # con_x = [conv(out) for conv in self.convs]
        # pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        # fc_x = torch.cat(pool_x, dim=1)
        # fc_x = fc_x.squeeze(-1)
        # fc_x = self.dropout(fc_x)
        # logit = self.fc(fc_x)
        # out = self.fc1(logit)
        return out
