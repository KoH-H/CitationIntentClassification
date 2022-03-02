# -*- coding: utf-8 -*-
# from utils.util import *
from train_valid.dataset_valid import *
import random
import copy
import collections
import torch.nn.functional as F
import torch.nn as nn
import itertools
from sklearn.cluster import KMeans



def sample_train(model, token, data, criterion, optimize, n_epoch, device, model_path=None):
    model.to(device=device)
    best_model_f1, counts, tmp,  = 0, 0, 0
    train_sen = data['train1']['sen']
    train_tar = data['train1']['tar']
    for i in range(n_epoch):
        model.train()
        avg_loss = 0
        tst = time.time()
        for index, (sen, tar) in enumerate(zip(train_sen, train_tar)):
            t_sent = token(sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            optimize.zero_grad()
            t_sent = t_sent.to(device)
            main_output = model(t_sent)
            loss = criterion(main_output, torch.LongTensor(tar).to(device))
            loss.backward()
            optimize.step()
            avg_loss += loss.item()
            if (index + 1) % 10 == 0:
                print('Batch: %d \t Loss: %.4f' % ((index + 1), avg_loss / 10))
                avg_loss = 0
        ted = time.time()
        print("Train time used:", ted - tst, i)
        val_f1, val_micro_f1 = dataset_valid(model, token, data['val'], device, criterion=criterion)
        print('Epoch: %d \t macro_f1: %.4f \t micro_f1: %.4f' % (i, val_f1, val_micro_f1))
        if val_f1 > best_model_f1:
            print('Val F1: %.4f > %.4f Saving Model' % (val_f1, best_model_f1))
            torch.save(model.state_dict(), model_path)
            best_model_f1 = val_f1
            counts = 1
            tmp = val_f1
    return best_model_f1


def multi_train(model, token, data, criterion, optimize, n_epoch, device, au_weight, model_path=None):
    model.to(device=device)
    best_model_f1, counts, tmp, best_epoch = 0, 0, 0, 0
    train_sen = data['train']['sen']
    train_tar = data['train']['tar']
    sec_sen = data['section']['sen']
    sec_tar = data['section']['tar']
    for i in range(n_epoch):
        model.train()
        avg_loss, avgori_loss, avgau_loss = 0, 0, 0
        train_ltrue_label, train_lpre_label = [], []
        tst = time.time()
        for index, (t_sen, t_tar, s_sen, s_tar) in enumerate(zip(train_sen, train_tar, sec_sen, sec_tar)):
            t_sent = token(t_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            s_sent = token(s_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            optimize.zero_grad()
            t_sent = t_sent.to(device)
            s_sent = s_sent.to(device)
            train_t_tar = torch.LongTensor(t_tar)
            s_tar = torch.LongTensor(s_tar)

            main_output, au_output = model(t_sent, au_x=s_sent)
            ori_loss = criterion(main_output, train_t_tar.to(device))
            au_loss = criterion(au_output, s_tar.to(device))
            loss = ori_loss + au_weight * au_loss
            loss.backward()
            optimize.step()
            avg_loss += loss.item()
            avgori_loss += ori_loss.item()
            avgau_loss += au_loss.item()

            pre_output = torch.softmax(main_output, dim=1)
            train_value, train_location = torch.max(pre_output, dim=1)
            train_lpre_label.extend(train_location.tolist())
            train_ltrue_label.extend(t_tar)

            if (index + 1) % 10 == 0:
                print('Batch: %d \t Loss: %.4f \t Avgori_loss: %.4f \t Avgau_loss: %.4f ' % (
                    (index + 1), avg_loss / 10, avgori_loss / 10, avgau_loss / 10))
                avg_loss, avgori_loss, avgau_loss = 0, 0, 0
        ted = time.time()
        print("Train time used:", ted - tst)
        l_macro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='macro')
        l_micro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='micro')
        print("Train".center(20, '='))
        print("Train L_macro_f1: %.4f \t L_micro_f1: %.4f " % (l_macro_f1, l_micro_f1))
        print("Train".center(20, '='))

        val_f1, val_micro_f1 = dataset_valid(model, token, data['val'], device, criterion=criterion)
        print("Val".center(20, '='))
        print('Epoch: %d \t macro_F1: %.4f \t micro_F1: %.4f' % (i, val_f1, val_micro_f1))
        print("Val".center(20, '='))

        if val_f1 > best_model_f1:
            print('Val F1: %.4f > %.4f Saving Model' % (val_f1, best_model_f1))
            torch.save(model.state_dict(), model_path)
            best_model_f1 = val_f1
            best_epoch = i
            counts = 1
            tmp = val_f1
    return best_model_f1, best_epoch


def rev_train(model, token, data, criterion, optimize, n_epoch, device, model_path=None, scheduler=None):
    model.to(device=device)
    best_model_f1, counts, tmp = 0, 0, 0
    train_sen = data['train']['sen']
    train_tar = data['train']['tar']
    re_sen = data['reverse']['sen']
    re_tar = data['reverse']['tar']
    for i in range(n_epoch):
        scheduler.step()
        model.train()
        avg_loss = 0
        tst = time.time()
        l = 1 - ((i - 1) / n_epoch) ** 2
        for index, (t_sen, t_tar, r_sen, r_tar) in enumerate(
                zip(train_sen, train_tar, re_sen, re_tar)):
            t_sent = token(t_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            r_sent = token(r_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            optimize.zero_grad()
            t_sent = t_sent.to(device)
            r_sent = r_sent.to(device)
            t_tar = torch.LongTensor(t_tar)
            r_tar = torch.LongTensor(r_tar)
            main_output = model(t_sent, r_sen=r_sent, l=l)
            loss = l * criterion(main_output, t_tar.to(device)) + (1 - l) * criterion(main_output, r_tar.to(device))
            loss.backward()
            optimize.step()
            avg_loss += loss.item()
            if (index + 1) % 10 == 0:
                print('Batch: %d \t Loss: %.4f' % ((index + 1), avg_loss / 10))
                avg_loss = 0
        ted = time.time()
        print("Train time used:", ted - tst, i)
        # val_f1 = trans_vail(modelacl, token, data['val'], device)
        val_f1, val_micro_f1 = dataset_valid(model, token, data['val'], device, criterion=criterion)
        print('Epoch: %d, F1: %.4f' % (i, val_f1))
        if val_f1 > best_model_f1:
            print('Val F1: %.4f > %.4f Saving Model' % (val_f1, best_model_f1))
            torch.save(model.state_dict(), model_path)
            best_model_f1 = val_f1
            counts = 1
            tmp = val_f1
    return best_model_f1



