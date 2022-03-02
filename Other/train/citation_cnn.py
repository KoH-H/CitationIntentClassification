# -*- coding: utf-8 -*-

from pathlib import Path
import torch
# from process_data import load_data
import torch.utils.data as Data
from model.model_l import LSTM, LSTMMULTIEMB
from model.model_lcn import LSTMCNN
from model.model_c import CNN
from model.model_la import LSTMAttention
from model.model_lac import BILSTMCNN
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
# from process_data import *
from load_data import *
import collections
import time
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from pgd_adversarial import PGD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.utils import *
from torch.distributions.beta import Beta
from lr_scheduler import WarmupMultiStepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def main():
    setup_seed(0)
    batch_size = 64
    hidden_size = 200
    lr = 0.0001
    n_epoch = 200
    best_val_f1 = 0
    s_train, s_val = load_train_val()
    s_test = load_test_data()
    # reverse_data = reverse_sampler(s_train)
    # unlabeled_data = load_unlabeled_data()
    # balanced_data = load_balanced_data()

    # normal
    vocab_glove = load_word_vector(s_train, s_val, s_test)
    # self_training
    # vocab_glove, vocab_fasttext = load_word_vector_unlabeled(s_train, s_val, s_test, unlabeled_data)
    # unlabeled_iter = generate_unlabeled_dataset(vocab_glove, unlabeled_data.index.values,
    #                                             unlabeled_data['citation_context'],
    #                                             unlabeled_data['citation_class_label'])
    # train_reverse = generate_reverse_train_dataset(vocab_glove, data_text=s_train['citation_context'],
    #                                                data_label=s_train['citation_class_label'],
    #                                                reverse_data_text=reverse_data['citation_context'],
    #                                                reverse_data_label=reverse_data['citation_class_label'])
    train_iter = generate_dataset(vocab_glove, s_train['citation_context'], s_train['citation_class_label'])
    val_iter = generate_dataset(vocab_glove, s_val['citation_context'], s_val['citation_class_label'])
    test_iter = generate_dataset(vocab_glove,  s_test['citation_context'], s_test['citation_class_label'])
    # balanced_iter = generate_dataset(vocab, balanced_data['citation_context'], balanced_data['citation_class_label'])
    train_iter = Data.DataLoader(train_iter, batch_size=batch_size, shuffle=False)
    val_iter = Data.DataLoader(val_iter, batch_size=batch_size, shuffle=False)

    test_iter = Data.DataLoader(test_iter, batch_size=batch_size, shuffle=False)
    vocab_size = vocab_glove.vectors.size()
    print('Total num. of words: {}, word vector dimension: {}'.format(vocab_size[0], vocab_size[1]))
    model_path = '/home/g19tka13/modelpth/cnn_model.pth'
    # LSTM
    # model1 = LSTM(vocab_glove, hidden_size=hidden_size, num_layers=2, batch=batch_size,
    #              bidirectional=True)

    # CNN
    model = CNN(vocab_glove, num_filters=128, filter_sizes=[2, 3, 4])

    # LSTM+CNN
    # model1 = LSTMCNN(vocab, hidden_size=hidden_size, num_layers=2, batch=batch_size,
    #                 bidirectional=True, filter_sizes=[2, 3, 4, 5], out_channels=125)
    # model1 = BILSTMCNN(vocab, hidden_size=hidden_size, num_layers=2,  bidirectional=True,
    #                   filter_sizes=[2, 3, 4, 5], out_channels=128)

    # LSTM+Attention
    # model1 = LSTMAttention(vocab, hidden_size=hidden_size, num_layers=2, batch=batch_size, bidirectional=True)
    # model1.embedding.weight.data = vocab.vectors
    # model1.embedding.weight.requires_grad = True
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device=device)
    ce_criterion = nn.CrossEntropyLoss(reduction='mean')
    # loss_fnc = nn.CosineEmbeddingLoss(reduction='mean', size_average=True, reduce=True)
    # loss_mse = nn.MSELoss(reduction='mean', size_average=True, reduce=True)
    for epoch in range(n_epoch):
        # model1.train放在哪参考网址 https://blog.csdn.net/andyL_05/article/details/107004401
        print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        model.train()
        avg_loss = 0
        # lamb = Beta(5, 5).sample()
        for item in train_iter:
            word_id, sen_len, t_class_id = item[0].to(device), item[1].to(device), item[2]
            optimizer.zero_grad()
            l2c_out = model(word_id=word_id)
            loss = ce_criterion(l2c_out, t_class_id.long().to(device))
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()

            # out = torch.softmax(l2c_out, dim=1)
            # _, train_y_pre = torch.max(out, 1)  # max函数有两个返回值(此处out是二维数组)第一个是最大值的list，第二个是值对应的位置
            # f1 = f1_score(t_class_id.long(), train_y_pre.cpu(), average='macro')
            print('Epoch: %d \t  Loss: %.4f' % (epoch, avg_loss))
        val_pre_label = []
        val_y_label = []
        model.eval()
        with torch.no_grad():
            for item in val_iter:
                v_word_id, v_sen_len, val_label = item[0], item[1], item[2]
                # v_sen_len = item[1]
                # val_label = item[2]
                v_word_id = v_word_id.to(device=device)
                v_sen_len = v_sen_len.to(device=device)
                out = model(word_id=v_word_id)
                out = torch.softmax(out, dim=1)
                _, val_y_pre = torch.max(out, 1)
                val_pre_label.extend(val_y_pre.cpu())
                val_y_label.extend(val_label)
        f1 = f1_score(torch.Tensor(val_y_label).long(), torch.Tensor(val_pre_label), average='macro')
        print(f1)
        if f1 > best_val_f1:
            print('Val Acc: %.4f > %.4f Saving model1' % (f1, best_val_f1))
            torch.save(model.state_dict(), model_path)
            best_val_f1 = f1
    test_pre_label = []
    test_y_label = []
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    model.eval()
    vector_list = []
    with torch.no_grad():
        for item_idx, item in enumerate(test_iter, 0):
            e_word_id, test_label = item[0],  item[2]
            e_word_id = e_word_id.to(device=device)
            # e_sen_len = e_sen_len.to(device=device)
            out = model(word_id=e_word_id)
            out = torch.softmax(out, dim=1)
            vector_list.append(out.cpu())
            _, test_pre = torch.max(out, 1)
            test_pre_label.extend(test_pre.cpu())
            test_y_label.extend(test_label)
    final_f1 = f1_score(torch.Tensor(test_y_label).long(), torch.Tensor(test_pre_label), average='macro')
    print('Test_pre_label', sorted(collections.Counter(torch.Tensor(test_pre_label).tolist()).items(), key=lambda x: x[0]))
    print('Test_y_label', sorted(collections.Counter(torch.Tensor(test_y_label).tolist()).items(), key=lambda x: x[0]))
    print('Test F1 : %.4f' % final_f1)
    generate_submission(torch.Tensor(test_pre_label).tolist(), 'cnn_result', final_f1)
    count = {}
    test_pre = torch.Tensor(test_pre_label).tolist()
    test_true = torch.Tensor(test_y_label).tolist()
    c_matrxi = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    print(c_matrxi)
    for i in range(len(test_true)):
        if test_true[i] == test_pre[i]:
            if test_true[i] not in count.keys():
                count[test_true[i]] = 1
            else:
                count[test_true[i]] = count[test_true[i]] + 1
    print(count)
    # pre_true = pd.DataFrame(columns=['true_id', 'pre_id'])
    # test_true_ser = pd.Series(test_true)
    # test_pre_ser = pd.Series(test_pre)
    # pre_true['true_id'] = test_true_ser
    # pre_true['pre_id'] = test_pre_ser
    # pre_true.to_csv('/home/g19tka13/taskA/true_predict.csv', sep=',', index=False)



if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time:', end_time - start_time)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))