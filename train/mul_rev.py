# -*- coding: utf-8 -*-
from transformers import AutoTokenizer
import sys
import torch.optim as optim
from model.based_bert import *
from utils.util import *
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from lr_sch import WarmupMultiStepLR
from datal_load import *
from train_method import *
from sklearn.metrics import classification_report
from train_valid.dataset_valid import *
start_time = time.time()
stop_words = ['et', 'al', 'e', 'g']
label = ['background', 'compares', 'extension', 'future', 'motivation', 'uses']

parser = argparse.ArgumentParser()
parser.add_argument("--cla", help="type of train", default=None, type=str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_rev(path, dev):
    print('Run main_rev')
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertRev('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    n_epoch = 140
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.01, warmup_epochs=5)
    data = load_data(batch_size=16, dataname='ACT', radio=0.8)
    best_model_f1 = rev_train(model, token, data, criterion, optimizer, 140, dev, model_path=path,
                              scheduler=scheduler)
    test_f1, test_true_label, test_pre_label = dataset_valid(model, token,
                                                          data['test'], device,
                                                          mode='test', path=path)
    print('Test_True_Label:', collections.Counter(test_true_label))
    print('Test_Pre_Label:', collections.Counter(test_pre_label))
    print('Test F1: %.4f Best Val F1: %.4f' % (test_f1, best_model_f1))
    test_true = torch.Tensor(test_true_label).tolist()
    test_pre = torch.Tensor(test_pre_label).tolist()
    generate_submission(test_pre, 'mul_rev_val_f1_{:.5}'.format(best_model_f1), test_f1, 'ACT')
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1, c_matrix, per_eval, lr=lr, epoch=n_epoch, fun_name='main_rev')


def main_mul(path, dev):  # two task
    print('Run main_mul')
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertMul('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    lr = 0.0005
    n_epoch = 80
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    data = load_data(batch_size=16, dataname='ACT', radio=0.8)
    best_model_f1 = multi_train(model, token, data, criterion, optimizer, 80, dev, au_weight=0.007,
                                model_path=path)
    test_f1, test_true_label, test_pre_label = dataset_valid(model, token,
                                                          data['test'], device,
                                                          mode='test', path=path)

    print('Test_True_Label:', collections.Counter(test_true_label))
    print('Test_Pre_Label:', collections.Counter(test_pre_label))
    print('Test F1: %.4f Best Val F1: %.4f' % (test_f1, best_model_f1))
    test_true = torch.Tensor(test_true_label).tolist()
    test_pre = torch.Tensor(test_pre_label).tolist()
    generate_submission(test_pre, 'mul_mul_val_f1_{:.5}'.format(best_model_f1), test_f1, 'ACT')
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1, c_matrix, lr=lr, epoch=n_epoch, fun_name='main_mul')


def main_sci(path, dev):
    print('Run main_mul')
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertBased('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    lr = 0.0005
    n_epoch = 80
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    data = load_data(batch_size=16, dataname='ACT', radio=0.8)
    best_model_f1 = sample_train(model, token, data, criterion, optimizer, 80, dev,
                                 model_path=path)
    test_f1, test_true_label, test_pre_label = dataset_valid(model, token,
                                                             data['test'], device,
                                                             mode='test', path=path)

    print('Test_True_Label:', collections.Counter(test_true_label))
    print('Test_Pre_Label:', collections.Counter(test_pre_label))
    print('Test F1: %.4f Best Val F1: %.4f' % (test_f1, best_model_f1))
    test_true = torch.Tensor(test_true_label).tolist()
    test_pre = torch.Tensor(test_pre_label).tolist()
    generate_submission(test_pre, 'mul_mul_val_f1_{:.5}'.format(best_model_f1), test_f1, 'ACT')
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1, c_matrix, lr=lr, epoch=n_epoch, fun_name='main_mul')



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.cla == 'mul':
        main_mul('mul_model.pth', device)
    elif args.cla == 'rev':
        main_rev('rev_model.pth', device)
    else:
        main_sci('scibert_model.pth', device)
