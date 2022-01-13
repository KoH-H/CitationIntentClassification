import jsonlines
import os
import pandas as pd
import argparse
# train_data = dict()
# label_list = list()
# with open('dataset/acl/train.jsonl', 'r+', encoding='utf8') as f:
#     for line in jsonlines.Reader(f):
#         label_list.append(line['intent'])
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
# train_data_df = pd.DataFrame(train_data)
# print(train_data_df)
# a = 'acl'
#
# if a == 'acl':
#     print(1234)
# print(label_list)

# label = list()
# with open('dataset/scicite/train.jsonl', 'r+', encoding='utf8') as f:
#     for line in jsonlines.Reader(f):
#         label.append(line['label'])
# print(label)

parse = argparse.ArgumentParser()
parse.add_argument("--dataset", help='name of data', type=str)
parse.add_argument("--test-test", type=str)
args = parse.parse_args()
print(args.dataset)
print(args.test_test)
