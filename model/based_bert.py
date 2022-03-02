# -*t coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import collections


class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class BertBased(nn.Module):
    def __init__(self, name):
        super(BertBased, self).__init__()
        self.model = AutoModel.from_pretrained(name)
        self.fc = nn.Linear(768, 6)
        self.drop = nn.Dropout(0.5)

        self.ori_word_att = nn.Linear(768, 384)
        self.ori_tanh = nn.Tanh()
        self.ori_word_weight = nn.Linear(384, 1, bias=False)

    def get_word(self, sen, bert_output, mask):
        input_t2n = sen['input_ids'].cpu().numpy()
        sep_location = np.argwhere(input_t2n == 103)  # 找到sep的位置
        sep_location = sep_location[:, -1]  # 每行查找的数字所在的位置
        select_index = list(range(sen['length'][0]))
        select_index.remove(0)  # 删除cls
        lhs = bert_output.last_hidden_state
        relength = []
        recomposing = []
        mask_recomposing = []
        for i in range(lhs.shape[0]):  # shape (batch, word_num, word_vector)
            select_index_f = select_index.copy()
            relength.append(sep_location[i] - 1)
            select_index_f.remove(sep_location[i])
            select_row = torch.index_select(lhs[i], 0,
                                            index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            select_mask = torch.index_select(mask[i], 0,
                                             index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            recomposing.append(select_row)
            mask_recomposing.append(select_mask)
        matrix = torch.stack(recomposing)
        mask = torch.stack(mask_recomposing)
        return matrix, mask

    def get_alpha(self, word_mat, mask):
        att_w = self.ori_word_att(word_mat)
        att_w = self.ori_tanh(att_w)
        att_w = self.ori_word_weight(att_w)

        mask = mask.unsqueeze(2)
        att_w = att_w.masked_fill(mask == 0, float('-inf'))
        att_w = F.softmax(att_w, dim=1)
        return att_w

    def get_sen_att(self, sen, bertouput, mask):
        word_mat, mask = self.get_word(sen, bertouput, mask)
        word_mat = self.drop(word_mat)  # embedding dropout
        att_w = self.get_alpha(word_mat, mask)
        word_mat = word_mat.permute(0, 2, 1)
        sen_pre = torch.bmm(word_mat, att_w).squeeze(2)
        return sen_pre

    # based for au
    def forward(self, x, **kwargs):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        bert_output = self.model(input_ids, attention_mask=attention_mask)
        sen_pre = self.get_sen_att(x, bert_output, attention_mask)
        bert_cls_hidden_state = self.fc(self.drop(sen_pre))
        return bert_cls_hidden_state


class BertMul(nn.Module):
    def __init__(self, name):
        super(BertMul, self).__init__()
        self.model = AutoModel.from_pretrained(name)
        self.fc = nn.Linear(768, 6)
        self.drop = nn.Dropout(0.5)
        self.au_fc = nn.Linear(768, 5)

        self.ori_word_atten = nn.Linear(768, 384)
        self.ori_tanh = nn.Tanh()
        self.ori_word_weight = nn.Linear(384, 1, bias=False)

    def get_word(self, sen, bert_output, mask):
        input_t2n = sen['input_ids'].cpu().numpy()
        sep_location = np.argwhere(input_t2n == 103)  # 找到sep的位置
        sep_location = sep_location[:, -1]  # 每行查找的数字所在的位置
        select_index = list(range(sen['length'][0]))
        select_index.remove(0)  # 删除cls
        lhs = bert_output.last_hidden_state
        relength = []
        recomposing = []
        mask_recomposing = []
        for i in range(lhs.shape[0]):  # shape (batch, word_num, word_vector)
            select_index_f = select_index.copy()
            relength.append(sep_location[i] - 1)
            select_index_f.remove(sep_location[i])
            select_row = torch.index_select(lhs[i], 0,
                                            index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            select_mask = torch.index_select(mask[i], 0,
                                             index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            recomposing.append(select_row)
            mask_recomposing.append(select_mask)
        matrix = torch.stack(recomposing)
        mask = torch.stack(mask_recomposing)
        return matrix, mask

    def get_alpha(self, word_mat, data_type, mask):
        if data_type == 'ori':
            att_w = self.ori_word_atten(word_mat)
            att_w = self.ori_tanh(att_w)
            att_w = self.ori_word_weight(att_w)
        else:
            att_w = self.au_word_atten(word_mat)
            att_w = self.au_tanh(att_w)
            att_w = self.au_word_weight(att_w)
        mask = mask.unsqueeze(2)
        att_w = att_w.masked_fill(mask == 0, float('-inf'))
        att_w = F.softmax(att_w, dim=1)
        return att_w

    def get_sen_att(self, sen, bert_ouput, data_type, mask):
        word_mat, mask = self.get_word(sen, bert_ouput, mask)
        word_mat = self.drop(word_mat)  # embedding dropout
        att_w = self.get_alpha(word_mat, data_type, mask)
        word_mat = word_mat.permute(0, 2, 1)
        sen_pre = torch.bmm(word_mat, att_w).squeeze(2)
        return sen_pre

    # multi
    def forward(self, x, au_x=None):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        bert_out = self.model(input_ids, attention_mask=attention_mask)
        ori_sen_pre = self.get_sen_att(x, bert_out, 'ori', attention_mask)

        if self.training:
            au_input_ids = au_x['input_ids']
            au_attention_mask = au_x['attention_mask']
            au_bert_out = self.model(au_input_ids, attention_mask=au_attention_mask)
            au_sen_pre = self.get_sen_att(au_x, au_bert_out, 'ori', au_attention_mask)

            au_re = self.au_fc(self.drop(au_sen_pre))
            ori_re = self.fc(self.drop(ori_sen_pre))
            return ori_re, au_re
        ori_re = self.fc(ori_sen_pre)
        return ori_re


class BertRev(nn.Module):
    def __init__(self, name):
        super(BertRev, self).__init__()

        self.model = AutoModel.from_pretrained(name)

        self.fc1 = nn.Linear(768 * 2, 768)
        self.fc = nn.Linear(768, 6)
        self.drop = nn.Dropout(0.5)

        self.ori_word_atten = nn.Linear(768, 384)
        self.ori_tanh = nn.Tanh()
        self.ori_word_weight = nn.Linear(384, 1, bias=False)

        self.re_word_atten = nn.Linear(768, 384)
        self.re_tanh = nn.Tanh()
        self.re_word_weight = nn.Linear(384, 1, bias=False)

    def get_alpha(self, word_mat, data_type, mask):
        if data_type == 'ori':
            # representation learning  attention
            att_w = self.ori_word_atten(word_mat)
            att_w = self.ori_tanh(att_w)
            att_w = self.ori_word_weight(att_w)
        else:
            # classification learning  attention
            att_w = self.re_word_atten(word_mat)
            att_w = self.re_tanh(att_w)
            att_w = self.re_word_weight(att_w)

        mask = mask.unsqueeze(2)
        att_w = att_w.masked_fill(mask == 0, float('-inf'))
        att_w = F.softmax(att_w, dim=1)
        return att_w

        # Get useful words vectors

    def get_word(self, sen, bert_output, mask):
        input_t2n = sen['input_ids'].cpu().numpy()
        sep_location = np.argwhere(input_t2n == 103)
        sep_location = sep_location[:, -1]
        select_index = list(range(sen['length'][0]))
        select_index.remove(0)  # 删除cls
        lhs = bert_output.last_hidden_state
        res = bert_output.hidden_states[8]
        relength = []
        recomposing = []
        mask_recomposing = []
        for i in range(lhs.shape[0]):
            select_index_f = select_index.copy()
            relength.append(sep_location[i] - 1)
            select_index_f.remove(sep_location[i])
            select_row = torch.index_select(lhs[i], 0,
                                            index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            select_mask = torch.index_select(mask[i], 0,
                                             index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            recomposing.append(select_row)
            mask_recomposing.append(select_mask)
        matrix = torch.stack(recomposing)
        mask = torch.stack(mask_recomposing)
        return matrix, mask

        # Get the representation vector after calculating the attention mechanism

    def get_sen_att(self, sen, bert_output, data_type, mask):
        word_mat, select_mask = self.get_word(sen, bert_output, mask)
        word_mat = self.drop(word_mat)
        att_w = self.get_alpha(word_mat, data_type, select_mask)
        word_mat = word_mat.permute(0, 2, 1)
        sen_pre = torch.bmm(word_mat, att_w).squeeze(2)
        return sen_pre
    # rev only
    def forward(self, x1, **kwargs):
        input_ids = x1['input_ids']
        attention_mask = x1['attention_mask']
        bert_output = self.model(input_ids, attention_mask=attention_mask)
        ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)

        if self.training:
            r_ids = kwargs['r_sen']['input_ids']
            r_attention_mask = kwargs['r_sen']['attention_mask']
            r_bert_output = self.model(r_ids, attention_mask=r_attention_mask)
            re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
            ori_sen_pre = self.drop(ori_sen_pre)
            re_sen_pre = self.drop(re_sen_pre)
            mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
            main_output = self.fc1(self.drop(mixed_feature))
            main_output = self.fc(main_output)

            return main_output
        re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
        mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
        mixed_feature = self.fc1(mixed_feature)
        mixed_feature = self.fc(mixed_feature)
        return mixed_feature
