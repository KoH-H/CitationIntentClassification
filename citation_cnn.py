# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
from model.model_lcn import LSTMCNN
from model.model_c import CNN
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score
import pandas as pd
from utils.load_data import *
from utils.util import *
import collections
import time
from sklearn.metrics import confusion_matrix, classification_report

# from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def main():
    setup_seed(0)
    batch_size = 64
    hidden_size = 200
    lr = 0.0001
    n_epoch = 200
    best_val_f1 = 0
    s_train, s_val, s_test  = load_train_val()
    # s_test = load_test_data()


    # normal
    vocab_glove = load_word_vector(s_train, s_val, s_test)
    train_iter = generate_dataset(vocab_glove, s_train['citation_context'], s_train['citation_class_label'])
    val_iter = generate_dataset(vocab_glove, s_val['citation_context'], s_val['citation_class_label'])
    test_iter = generate_dataset(vocab_glove,  s_test['citation_context'], s_test['citation_class_label'])
    # balanced_iter = generate_dataset(vocab, balanced_data['citation_context'], balanced_data['citation_class_label'])
    train_iter = Data.DataLoader(train_iter, batch_size=batch_size, shuffle=False)
    val_iter = Data.DataLoader(val_iter, batch_size=batch_size, shuffle=False)

    test_iter = Data.DataLoader(test_iter, batch_size=batch_size, shuffle=False)
    vocab_size = vocab_glove.vectors.size()
    print('Total num. of words: {}, word vector dimension: {}'.format(vocab_size[0], vocab_size[1]))
    model_path = 'cnn_model.pth'

    model = CNN(vocab_glove, num_filters=128, filter_sizes=[2, 3, 4])

    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device=device)
    ce_criterion = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(n_epoch):
        # print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        model.train()
        for item in train_iter:
            word_id, sen_len, t_class_id = item[0].to(device), item[1].to(device), item[2]
            optimizer.zero_grad()
            l2c_out = model(word_id=word_id)
            loss = ce_criterion(l2c_out, t_class_id.long().to(device))
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()

            print('Epoch: %d \t  Loss: %.4f' % (epoch, avg_loss))
        val_pre_label = []
        val_y_label = []
        model.eval()
        with torch.no_grad():
            for item in val_iter:
                v_word_id, v_sen_len, val_label = item[0], item[1], item[2]
                v_word_id = v_word_id.to(device=device)
                # v_sen_len = v_sen_len.to(device=device)
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
    # generate_submission(torch.Tensor(test_pre_label).tolist(), 'cnn_result', final_f1)
    count = {}
    test_pre = torch.Tensor(test_pre_label).tolist()
    test_true = torch.Tensor(test_y_label).tolist()
    c_matrxi = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    print(c_matrxi)
    for i in range(len(test_true)):
        if test_true[i] == test_pre[i]:
            if test_true[i] not in count.keys():
                count[test_true[i]] = 1
            else:
                count[test_true[i]] = count[test_true[i]] + 1
    print(count)
    log_result(final_f1, 0, c_matrxi, per_eval, lr=lr, epoch=n_epoch, fun_name='textcnn')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time:', end_time - start_time)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))