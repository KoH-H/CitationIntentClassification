import jsonlines
import os
import pandas as pd
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
print(123)
os.system("tar -zxvf dataset/acl/acl.tar.gz -C dataset/acl/")