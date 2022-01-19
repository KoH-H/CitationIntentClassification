import jsonlines
import os
import pandas as pd
import argparse
from utils.dataload import scijson2pd
# train_data = dict()
# label_list = list()
a = 0
# os.system("tar -zxvf dataset/acl/acl.tar.gz -C dataset/acl")
with open('dataset/acl/train.jsonl', 'r+', encoding='utf8') as f:
    for line in jsonlines.Reader(f):
        a = a + 1
        # label_list.append(line['intent'])
        # print(line['intent'])
        # exit()
        # if 'citation_context' not in train_data:
        #     train_data['citation_context'] = [line['text']]
        #     train_data['citation_class_label'] = [line['intent']]
        # else:
        #     context_list = train_data['citation_context']
        #     context_list.append(line['text'])
        #     label_list = train_data['citation_class_label']
        #     label_list.append(line['intent'])
        #     train_data['citation_context'] = context_list
        #     train_data['citation_class_label'] = label_list
print(a)
# train_data_df = pd.DataFrame(train_data)
# print(train_data_df)
# a = 'acl'
#
# if a == 'acl':
#     print(1234)
# print(label_list)

# total = 0
# back = 0
# method = 0
# result = 0
# with open('dataset/scicite/test.jsonl', 'r+', encoding='utf8') as f:
#     for line in jsonlines.Reader(f):
#         total = total + 1
#         if line['label'] == 'background':
#             back = back + 1
#         elif line['label'] == 'method':
#             method = method + 1
#         else:
#             result = result + 1
# print(back, method, result, total)

# parse = argparse.ArgumentParser()
# parse.add_argument("--dataset", help='name of data', type=str)
# parse.add_argument("--test-test", type=str)
# args = parse.parse_args()
# print(args.dataset)
# print(args.test_test)
result = scijson2pd(['train', 'dev', 'test'], [20, 10, 1])
print(result['train'].shape[0])
print(result['dev'].shape[0])
print(result['test'].shape[0])
