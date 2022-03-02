# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os.path
from pathlib import Path
import nltk
from torchtext.vocab import Vocab, Vectors
import re
import collections
import torch.utils.data as Data
import torch
import zipfile
import sklearn
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.corpus import stopwords


stop_words = stopwords.words('english')
stop_words = ['et', 'al.', 'e', 'g', 'i.e.', 'e.g.', 'al']
data_path = Path('/home/g19tka13/taskA')
label = ['background', 'compares', 'extension', 'future', 'motivation', 'uses']
stop_words = stop_words + label
class_num = 6


def load_train_val():
    """
    load train_me val data
    :return: the content of data str to list
    """
    # np.random.seed(rand_number)
    print('Loading train_me data')
    # taskA_train_data = data_path / 'train_translate_balanced.csv'
    # taskA_train_data = data_path / 'train_translate.csv'
    taskA_train_data = data_path / 'SDP_train.csv'
    df = pd.read_csv(taskA_train_data, sep=',')
    class_num = df['citation_class_label'].value_counts()
    print('class_num \n', class_num)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df = df[(df['citation_class_label'] == 3) | (df['citation_class_label'] == 0)]
    df = sklearn.utils.shuffle(df, random_state=0).reset_index(drop=True)  # random seed
    print(df)
    # df = df.head(300)
    total_instance_num = float(df.shape[0])
    print(df.shape)
    df_header = df.columns
    train_data = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
    reverse_train_data = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
    class_num = collections.Counter(df['citation_class_label'])  # Counter.items  Counter 结果未进行排序
    class_id_sort = sorted(list(class_num.keys()))
    # weighted = np.zeros(6, dtype=np.float32)
    # for i in class_id_sort:
    #     weighted[i] = np.log10((total_instance_num - class_num[i])/total_instance_num) + 1
    # class_num = df['citation_class_label'].value_counts()
    # print('class_num \n', class_num)
    # weighted = np.zeros(6, dtype=np.float32)
    # 保留小数位数 使用%.2f，round函数或者float函数。假设要保留两位小数
    # round(num, 2) || '%.2f' % num || float('%.2f' % num)
    # for i in range(6):
    #     weighted[i] = np.log10((total_instance_num - class_num[i]) / class_num[i])
    for index, row in df.iterrows():
        cited_author = row['cited_author']
        citation_text = re.sub(r"#AUTHOR_TAG", cited_author, row['citation_context'])
        # citation_text = re.sub(r'[\[0-9\]]', '', citation_text).lower()
        # citation_text = re.sub(r' \(.*?\)',  '', citation_text)
        citation_text = re.sub(r'[^a-zA-Z]', ' ', citation_text).lower()  # 删除无用的数据如何数字 标点符号之类
        # citation_text = re.sub(r'[0-9]|[\[0-9\]]', '', citation_text).lower()
        citation_text = nltk.word_tokenize(citation_text)
        citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
        # citation_word = []
        # ner_list = []
        # for word in citation_text:
        #     if word not in stop_words and len(word) > 1:
        #         if word == 'authortag':
        #             citation_word.append(row['cited_author'].lower())
        #             ner_list.append(1)
        #             continue
        #         citation_word.append(word)
        #         ner_list.append(0)
        # ner.append(ner_list)
        # train_data.loc[index] = {"unique_id": row['unique_id'],
        #                          'core_id': row['core_id'],
        #                          'citing_title': row['citing_title'],
        #                          'citing_author': row['citing_author'],
        #                          'cited_title': row['cited_title'],
        #                          'cited_author': row['cited_author'],
        #                          'citation_context': citation_text,
        #                          'citation_class_label': row['citation_class_label']}
        # train_length_list.append(len(citation_text))
        # if len(citation_text) > 50:
        #     citation_text = citation_text[:50]
        # if len(citation_text) < 5:
        #     train_data.loc[index] = {"citation_context": np.nan,
        #                              "citation_class_label": row['citation_class_label']}
        #     continue
        # if len(citation_text) > 50:
        #     citation_text = citation_text[:50]
        train_data.loc[index] = {"citation_context": citation_text,
                                 "citation_class_label": row['citation_class_label']}
        # if index < int(df.shape[0] * 0.8):
        #     citation_text.reverse()
        #     reverse_train_data.loc[index] = {"citation_context": citation_text,
        #                                      "citation_class_label": row['citation_class_label']}
        # print(citation_text)
    train_data = train_data.dropna(axis=0, how='any').reset_index(drop=True)
    train = train_data.loc[:int(train_data.shape[0] * 0.8) - 1]
    val = (train_data.loc[int(train_data.shape[0] * 0.8):]).reset_index(drop=True)
    # train_me = train_me.append(reverse_train_data, ignore_index=True)
    print(10 * '=', "real_train", 10 * '=')
    print(train['citation_class_label'].value_counts())
    print(10 * '=', "real_train", 10 * '=')
    # display sentences length and num
    # ctrain = collections.Counter(train_length_list)
    # length_sort = sorted(list(ctrain.keys()))
    # print('train_length', length_sort)
    # num_list = []
    # for i in length_sort:
    #     num_list.append(ctrain[i])
    # print(num_list)
    # print(ctrain)
    # plt.figure(figsize=(30, 6), dpi=100)
    # mpl.rcParams['font.family'] = 'SimHei'
    # length_sort = [str(length) for length in length_sort]
    # plt.bar(length_sort, num_list, width=0.5, color='red')
    # plt.grid(alpha=0.3, linestyle=':')
    # plt.xlabel('sentences length')
    # plt.ylabel('sentences num')
    # plt.show()
    return train, val


def reverse_sampler(traindata):
    random.seed(0)
    counter_train = dict(traindata['citation_class_label'].value_counts())
    num_list = list(counter_train.values())
    class_list = list(counter_train.keys())
    max_num = max(num_list)
    class_weight = [max_num / i for i in num_list]
    sum_weight = sum(class_weight)
    class_dict = dict()
    # print(traindata[traindata['citation_class_label'] == 0].index.values.tolist())
    sampled_examples = []
    for i in class_list:
        class_dict[i] = traindata[traindata['citation_class_label'] == i].index.values.tolist()
    total_samples = sum(num_list)
    for _ in range(total_samples):
        rand_number, now_sum = random.random() * sum_weight, 0
        for j in class_list:
            now_sum += class_weight[class_list.index(j)]
            if rand_number <= now_sum:
                sampled_examples.append(random.choice(class_dict[j]))
                break
    print('reverse sample count{}'.format(collections.Counter(traindata.iloc[sampled_examples, :]['citation_class_label'])))
    # print(collections.Counter(sampled_examples))
    revere_data = traindata.iloc[sampled_examples, :].reset_index(drop=True)   # 不reset_index的话因为重复采样会出现多个相同的索引
    revere_data.loc[:, 'change'] = 0
    return revere_data


def load_test_data():
    """
    :return:
    """
    print('Loading test data')
    taskA_test_data = data_path / 'SDP_test.csv'
    test_df = pd.read_csv(taskA_test_data, sep=',').merge(pd.read_csv(str(taskA_test_data).replace
                                                                      ('SDP_test', 'sample_submission')), on='unique_id')
    # test_df = test_df.sample(frac=1).reset_index(drop=True)
    # test_df = test_df.head(500)
    print(test_df.shape)
    test_df_header = test_df.columns
    test_data = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
    for index, row in test_df.iterrows():
        label_word = str(label[row['citation_class_label']])
        cited_author = row['cited_author']
        citation_text = re.sub(r"#AUTHOR_TAG", cited_author, row['citation_context'])
        # citation_text = re.sub(r'[\[0-9\]]', '', citation_text).lower()
        # citation_text = re.sub(r' \(.*?\)', '', citation_text)
        citation_text = re.sub(r'[^a-zA-Z]', ' ', citation_text).lower()
        # citation_text = re.sub(r'[0-9]|[\[0-9\]]', '', citation_text).lower()
        citation_text = nltk.word_tokenize(citation_text)
        citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
        # test_data.loc[index] = {"unique_id": row['unique_id'],
        #                         'core_id': row['core_id'],
        #                         'citing_title': row['citing_title'],
        #                         'citing_author': row['citing_author'],
        #                         'cited_title': row['cited_title'],
        #                         'cited_author': row['cited_author'],
        #                         'citation_context': citation_text,
        #                         'citation_class_label': row['citation_class_label']}
        # test_length_list.append(len(citation_text))
        # if len(citation_text) > 50:
        #     citation_text = citation_text[:50]
        # if len(citation_text) < 5:
        #     test_data.loc[index] = {'citation_context': np.nan,
        #                             'citation_class_label': row['citation_class_label']}
        #     continue
        # if len(citation_text) > 50:
        #     citation_text = citation_text[: 50]
        test_data.loc[index] = {'citation_context': citation_text,
                                'citation_class_label': row['citation_class_label']}
    # display sentences length and num
    # print('test_length', collections.Counter(test_length_list))
    # citation_counter = collections.Counter(test_length_list)
    # length_list = sorted(list(citation_counter.keys()))
    # num_list = []
    # for i in length_list:
    #     num_list.append(citation_counter[i])
    # length_list = [str(length) for length in length_list]
    # plt.figure(figsize=(30, 6), dpi=100)
    # mpl.rcParams['font.family'] = 'SimHei'
    # # length_sort = [str(length) for length in length_sort]
    # plt.bar(length_list, num_list, width=0.5, color='red')
    # plt.grid(alpha=0.3, linestyle=':')
    # plt.xlabel('test sentences length')
    # plt.ylabel('test sentences num')
    # plt.show()
    test_data = test_data.dropna(axis=0, how='any').reset_index(drop=True)
    return test_data


def load_balanced_data():
    """
    :return:
    """
    print('Loading balanced data')
    taskA_balanced_data = data_path / 'data_balanced.csv'
    balanced_df = pd.read_csv(taskA_balanced_data, sep=',')
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    # test_df = test_df.head(500)
    print(balanced_df.shape)
    test_df_header = balanced_df.columns
    balanced_data = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
    for index, row in balanced_df.iterrows():
        label_word = str(label[row['citation_class_label']])
        cited_author = row['cited_author']
        citation_text = re.sub(r"#AUTHOR_TAG", cited_author, row['citation_context'])
        # citation_text = re.sub(r'[\[0-9\]]', '', citation_text).lower()
        # citation_text = re.sub(r' \(.*?\)', '', citation_text)
        citation_text = re.sub(r'[^a-zA-Z]', ' ', citation_text).lower()
        # citation_text = re.sub(r'[0-9]|[\[0-9\]]', '', citation_text).lower()
        citation_text = nltk.word_tokenize(citation_text)
        citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
        # test_data.loc[index] = {"unique_id": row['unique_id'],
        #                         'core_id': row['core_id'],
        #                         'citing_title': row['citing_title'],
        #                         'citing_author': row['citing_author'],
        #                         'cited_title': row['cited_title'],
        #                         'cited_author': row['cited_author'],
        #                         'citation_context': citation_text,
        #                         'citation_class_label': row['citation_class_label']}
        # test_length_list.append(len(citation_text))
        # if len(citation_text) > 50:
        #     citation_text = citation_text[:50]
        # if len(citation_text) < 5:
        #     test_data.loc[index] = {'citation_context': np.nan,
        #                             'citation_class_label': row['citation_class_label']}
        #     continue
        balanced_data.loc[index] = {'citation_context': citation_text,
                                    'citation_class_label': row['citation_class_label']}
    # display sentences length and num
    # print('test_length', collections.Counter(test_length_list))
    # citation_counter = collections.Counter(test_length_list)
    # length_list = sorted(list(citation_counter.keys()))
    # num_list = []
    # for i in length_list:
    #     num_list.append(citation_counter[i])
    # length_list = [str(length) for length in length_list]
    # plt.figure(figsize=(30, 6), dpi=100)
    # mpl.rcParams['font.family'] = 'SimHei'
    # # length_sort = [str(length) for length in length_sort]
    # plt.bar(length_list, num_list, width=0.5, color='red')
    # plt.grid(alpha=0.3, linestyle=':')
    # plt.xlabel('test sentences length')
    # plt.ylabel('test sentences num')
    # plt.show()
    balanced_data = balanced_data.dropna(axis=0, how='any').reset_index(drop=True)
    return balanced_data


def load_unlabeled_data():
    print('Loading Unlabeled data')
    unlabeled_path = data_path / 'aclgenerate.csv'
    data = pd.read_csv(unlabeled_path, sep=',')
    # print(data.head(3000))
    used_data = data.head(3000)
    unlabeled_data = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
    for index, row in used_data.iterrows():
        citation_context = re.sub(r'[^a-zA-Z]', ' ', row['citation_context']).lower()
        citation_text = nltk.word_tokenize(citation_context)
        citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
        unlabeled_data.loc[index] = {'citation_context': citation_text, 'citation_class_label': 0}
    # print(unlabeled_data)
    return unlabeled_data

def count_all_words(data):
    words = []
    for row in data:
        words.extend(row)
    return words


def load_word_vector(train_data, val_data, test_data, used_unlabeled_data=None):
    """
    :param train_data:
    :param val_data:
    :param test_data:
    :param train_type:
    :param used_unlabeled_data:
    :return:
    """
    label_dataframe = pd.Series(label)
    # Download word vector
    print('Loading word vectors')
    path = os.path.join('/home/g19tka13/wordvector', 'glove.6B.zip')
    unzip_path = Path('/home/g19tka13/wordvector')
    if not os.path.exists(path):
        print('Download word vectors')
        import urllib.request
        urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip',
                                   path)
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall(unzip_path)
    vectors_glove = Vectors('glove.6B.300d.txt', cache='/home/g19tka13/wordvector')
    # vectors_fasttext = Vectors('wiki.en.vec', cache='/home/g19tka13/wordvector')

    vocab_glove = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append
                                                      (val_data['citation_context'], ignore_index=True).append
                                                      (test_data['citation_context'], ignore_index=True).append
                                                      (label_dataframe, ignore_index=True))),
                        specials=['<pad>', '<unk>'], vectors=vectors_glove)
    # vocab_fasttext = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append
    #                                                         (val_data['citation_context'], ignore_index=True).append
    #                                                         (test_data['citation_context'],
    #                                                          ignore_index=True).append
    #                                                         (label_dataframe, ignore_index=True))),
    #                        specials=['<pad>', '<unk>'], vectors=vectors_fasttext)
    return vocab_glove


def load_word_vector_unlabeled(train_data, val_data, test_data, unlabeled_data=None):
    """
    :param train_data:
    :param val_data:
    :param test_data:
    :param train_type:
    :param used_unlabeled_data:
    :return:
    """
    label_dataframe = pd.Series(label)
    # Download word vector
    print('Loading word vectors')
    path = os.path.join('/home/g19tka13/wordvector', 'glove.6B.zip')
    unzip_path = Path('/home/g19tka13/wordvector')
    if not os.path.exists(path):
        print('Download word vectors')
        import urllib.request
        urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip',
                                   path)
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall(unzip_path)
    vectors_glove = Vectors('glove.6B.300d.txt', cache='/home/g19tka13/wordvector')
    vectors_fasttext = Vectors('wiki.en.vec', cache='/home/g19tka13/wordvector')

    vocab_glove = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append
                                           (val_data['citation_context'], ignore_index=True).append
                                           (test_data['citation_context'], ignore_index=True).append
                                           (unlabeled_data['citation_context'], ignore_index=True).append
                                           (label_dataframe, ignore_index=True))),
                        specials=['<pad>', '<unk>'], vectors=vectors_glove)
    vocab_fasttext = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append
                                              (val_data['citation_context'], ignore_index=True).append
                                              (test_data['citation_context'], ignore_index=True).append
                                              (unlabeled_data['citation_context'], ignore_index=True).append
                                              (label_dataframe, ignore_index=True))),
                           specials=['<pad>', '<unk>'], vectors=vectors_fasttext)
    return vocab_glove, vocab_fasttext


def default_generate(vocabulary, data_text, data_label, **kwargs):
    text_data = data_text
    train_text_num = len(text_data)
    text_len = np.array([len(sentence) for sentence in text_data])
    max_text_len = max(text_len)
    word_to_id = vocabulary.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
    for i in range(len(text_data)):
        word_to_id[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
                                             vocabulary.stoi['<unk>'] for word in text_data[i]]
    return word_to_id, text_len


def generate_unlabeled_dataset(vocabulary, index, data_text, data_label):
    word_id, text_len = default_generate(vocabulary, data_text, data_label)
    dataset = Data.TensorDataset(torch.from_numpy(word_id), torch.from_numpy(text_len), torch.from_numpy(index))
    return dataset


def generate_dataset(vocabulary, data_text=None, data_label=None, ner_class = None, **kwargs):
    """
    将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
    同时将sentence和label和sentence_len组装起来方便载入。
    :param data_text:
    :param vocabulary:
    :param data_label:
    :return:
    """
    # if data_type == 'train_me':
    text_data = data_text
    train_text_num = len(text_data)
    text_len = np.array([len(sentence) for sentence in text_data])
    max_text_len = max(text_len)
    # word_to_index
    word_to_id = vocabulary.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
    for i in range(len(text_data)):
        # if len(kwargs) != 0:
        #     print(len((text_data[i])))
        # for word in text_data[i]:
        #     print(word)
        word_to_id[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
                                             vocabulary.stoi['<unk>'] for word in text_data[i]]
    # iter = (word_id, sentence_len, class_id)
    # if len(kwargs) != 0:
    #     dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
    #                                  torch.from_numpy(kwargs['index']))
    # else:
    label_data = data_label.values.astype(np.int64)
        # dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
        #                              torch.from_numpy(label_data))
        # if ner_class is not None:
        #     dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
        #                                  torch.from_numpy(label_data), ner_class)
        # else:
    dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
                                 torch.from_numpy(label_data))
    return dataset


# def generate_dataset_multi_emb(vocabulary, vocabulary_fasttext, data_text=None, data_label=None, ner_class = None, **kwargs):
#     """
#     将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
#     同时将sentence和label和sentence_len组装起来方便载入。
#     :param data_text:
#     :param vocabulary:
#     :param data_label:
#     :return:
#     """
#     # if data_type == 'train_me':
#     text_data = data_text
#     train_text_num = len(text_data)
#     text_len = np.array([len(sentence) for sentence in text_data])
#     max_text_len = max(text_len)
#     # word_to_index
#     word_to_id = vocabulary.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
#     word_to_id_fasttext = vocabulary_fasttext.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
#     for i in range(len(text_data)):
#         # if len(kwargs) != 0:
#         #     print(len((text_data[i])))
#         # for word in text_data[i]:
#         #     print(word)
#         word_to_id[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
#                                              vocabulary.stoi['<unk>'] for word in text_data[i]]
#         word_to_id_fasttext[i, :len(text_data[i])] = [vocabulary_fasttext.stoi[word] if word in vocabulary_fasttext.stoi else
#                                                       vocabulary_fasttext.stoi['<unk>'] for word in text_data[i]]
#         # iter = (word_id, sentence_len, class_id)
#     # iter = (word_id, sentence_len, class_id)
#     if len(kwargs) != 0:
#         dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                      torch.from_numpy(kwargs['index']), torch.from_numpy(word_to_id_fasttext))
#     else:
#         label_data = data_label.values.astype(np.int64)
#         # dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#         #                              torch.from_numpy(label_data))
#         if ner_class is not None:
#             dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                          torch.from_numpy(label_data), ner_class, torch.from_numpy(word_to_id_fasttext))
#         else:
#             dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                          torch.from_numpy(label_data), torch.from_numpy(word_to_id_fasttext))
#     return dataset


# def generate_reverse_train_dataset_multi_emb(vocabulary, vocabulary_fastext, data_text=None, data_label=None,
#                                    reverse_data_text=None, reverse_data_label=None, ner_class = None, **kwargs):
#     """
#     将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
#     同时将sentence和label和sentence_len组装起来方便载入。
#     :param data_text:
#     :param vocabulary:
#     :param data_label:
#     :return:
#     """
#     text_data = data_text
#     reverse_data = reverse_data_text
#     train_text_num = len(text_data)
#     reverse_text_num = len(reverse_data)
#     text_len = np.array([len(sentence) for sentence in text_data])
#     reverse_text_len = np.array([len(sentence) for sentence in reverse_data])
#     max_text_len = max(text_len)
#     reverse_max_test_len = max(reverse_text_len)
#     # word_to_index
#     word_to_id = vocabulary.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
#     reverse_to_id = vocabulary.stoi['<pad>'] * np.ones([reverse_text_num, reverse_max_test_len], dtype=np.int64)
#     word_to_id_fasttext = vocabulary_fastext.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
#     reverse_to_id_fasttext = vocabulary_fastext.stoi['<pad>'] * np.ones([reverse_text_num, reverse_max_test_len], dtype=np.int64)
#     for i in range(len(text_data)):
#         # if len(kwargs) != 0:
#         #     print(len((text_data[i])))
#         # for word in text_data[i]:
#         #     print(word)
#         word_to_id[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
#                                              vocabulary.stoi['<unk>'] for word in text_data[i]]
#         word_to_id_fasttext[i, :len(text_data[i])] = [vocabulary_fastext.stoi[word] if word in vocabulary_fastext.stoi
#                                                       else vocabulary_fastext.stoi['<unk>'] for word in text_data[i]]
#     # iter = (word_id, sentence_len, class_id)
#     for j in range(len(reverse_data)):
#         reverse_to_id[j, :len(reverse_data[j])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
#                                                    vocabulary.stoi['<unk>'] for word in reverse_data[j]]
#         reverse_to_id_fasttext[j, :len(reverse_data[j])] = [vocabulary_fastext.stoi[word] if word in vocabulary_fastext.stoi
#                                                             else vocabulary_fastext.stoi['<unk>'] for word in reverse_data[j]]
#     if len(kwargs) != 0:
#         dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                      torch.from_numpy(kwargs['index']))
#     else:
#         label_data = data_label.values.astype(np.int64)
#         print(type(label_data))
#         reverse_label_data = reverse_data_label.values.astype(np.int64)
#         print(type(reverse_label_data))
#         # dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#         #                              torch.from_numpy(label_data))
#         if ner_class is not None:
#             dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                          torch.from_numpy(label_data), ner_class, torch.from_numpy(reverse_to_id),
#                                          torch.from_numpy(reverse_text_len), torch.from_numpy(reverse_label_data),
#                                          torch.from_numpy(word_to_id_fasttext), torch.from_numpy(reverse_to_id_fasttext))
#         else:
#             dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
#                                          torch.from_numpy(label_data), torch.from_numpy(reverse_to_id),
#                                          torch.from_numpy(reverse_text_len), torch.from_numpy(reverse_label_data),
#                                          torch.from_numpy(word_to_id_fasttext), torch.from_numpy(reverse_to_id_fasttext))
#     return dataset


def generate_reverse_train_dataset(vocabulary, data_text=None, data_label=None,
                                   reverse_data_text=None, reverse_data_label=None, ner_class = None, **kwargs):
    """
    将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
    同时将sentence和label和sentence_len组装起来方便载入。
    :param data_text:
    :param vocabulary:
    :param data_label:
    :return:
    """
    text_data = data_text
    reverse_data = reverse_data_text
    train_text_num = len(text_data)
    reverse_text_num = len(reverse_data)
    text_len = np.array([len(sentence) for sentence in text_data])
    reverse_text_len = np.array([len(sentence) for sentence in reverse_data])
    max_text_len = max(text_len)
    reverse_max_test_len = max(reverse_text_len)
    # word_to_index
    word_to_id = vocabulary.stoi['<pad>'] * np.ones([train_text_num, max_text_len], dtype=np.int64)
    reverse_to_id = vocabulary.stoi['<pad>'] * np.ones([reverse_text_num, reverse_max_test_len], dtype=np.int64)
    for i in range(len(text_data)):
        # if len(kwargs) != 0:
        #     print(len((text_data[i])))
        # for word in text_data[i]:
        #     print(word)
        word_to_id[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
                                             vocabulary.stoi['<unk>'] for word in text_data[i]]
    # iter = (word_id, sentence_len, class_id)
    for j in range(len(reverse_data)):
        reverse_to_id[j, :len(reverse_data[j])] = [vocabulary.stoi[word] if word in vocabulary.stoi else
                                                   vocabulary.stoi['<unk>'] for word in reverse_data[j]]
    if len(kwargs) != 0:
        dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
                                     torch.from_numpy(kwargs['index']))
    else:
        label_data = data_label.values.astype(np.int64)
        print(type(label_data))
        reverse_label_data = reverse_data_label.values.astype(np.int64)
        print(type(reverse_label_data))
        # dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
        #                              torch.from_numpy(label_data))
        if ner_class is not None:
            dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
                                         torch.from_numpy(label_data), ner_class, torch.from_numpy(reverse_to_id),
                                         torch.from_numpy(reverse_text_len), torch.from_numpy(reverse_label_data))
        else:
            dataset = Data.TensorDataset(torch.from_numpy(word_to_id), torch.from_numpy(text_len),
                                         torch.from_numpy(label_data), torch.from_numpy(reverse_to_id),
                                         torch.from_numpy(reverse_text_len), torch.from_numpy(reverse_label_data))
    return dataset

def generate_li(vocabulary):
    """
    generate label to id
    :param vocabulary:
    :return:
    """
    label_to_id = vocabulary.stoi['<pad>'] * np.ones([6, 1], dtype=np.int64)
    for i in range(6):
        label_to_id[i, :1] = [vocabulary.stoi[label[i]] if label[i] in vocabulary.stoi else vocabulary.stoi['<unk>']]

    return torch.LongTensor(label_to_id)


if __name__ == '__main__':
    # train_me, val, weighted = load_train_val()
    # reverse_sampler(train_me)
    load_unlabeled_data()
