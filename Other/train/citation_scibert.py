# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import sys
sys.path.append('/home/g19tka13/3c')
import torch.optim as optim
from transformer_model.model.based_bert import *
from lr_scheduler import WarmupMultiStepLR
from transformer_model.dataloader.data_load import *
from transformer_model.train_me.train_method import *
from utils.utils import *
from sklearn.metrics import classification_report
import optuna


def scibert_m(path, dev):
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # model1 = BertBased('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    # lr = 0.0005
    n_epoch = 80
    # optimizer = optim.SGD(model1.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    # scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.01, warmup_epochs=5)
    data = multi_load_data(16)

    def objective(trial):
        model = BertBased('allenai/scibert_scivocab_uncased')
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        best_model_f1 = sample_train(model, token, data, criterion, optimizer, n_epoch, dev, model_path=path)
    # model_state = torch.load(path)
    # model1.load_state_dict(model_state)
    # model1.to(device)
        test_f1, test_micro_f1, test_true_label, test_pre_label = trans_vail(model, token,
                                                          data['test'], device,
                                                          mode='test', path=path)

        print('Test_True_Label:', collections.Counter(test_true_label))
        print('Test_Pre_Label:', collections.Counter(test_pre_label))
        print('Test F1: %.4f \t Test micro_f1: %.4f \t Best Val F1: %.4f' % (test_f1, test_micro_f1, best_model_f1))
        test_true = torch.Tensor(test_true_label).tolist()
        test_pre = torch.Tensor(test_pre_label).tolist()
        generate_submission(test_pre, 'scibert', test_f1)
        c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
        per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
        log_result(test_f1, best_model_f1, c_matrix, per_eval, lr=lr, epoch=n_epoch, fun_name='scibert')
        return best_model_f1
    study = optuna.create_study(study_name='scibertatt', direction='maximize', storage='sqlite:///scibertatt.db')
    study.optimize(objective, n_trials=10)
    print("Best_Params:{} \t Best_Value:{}".format(study.best_params, study.best_value))
    history = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(history)
    # print(c_matrix)
    # print("mul", lr)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    scibert_m('/home/g19tka13/modelpth/scibert_model.pth', device)