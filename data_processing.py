# -*- coding: utf-8 -*-
import re

import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import json
import collections
import os

# os.system("tar -zxvf dataset/acl/acl.tar.gz -C dataset/acl/")
train_path = Path('dataset/acl/train.jsonl')
dev_path = Path('dataset/acl/dev.jsonl')
test_path = Path('dataset/acl/test.jsonl')
# section = root_path / 'sections-scaffold-train1.jsonl'

core_id = []
citing_title = []
citing_author = []
cited_title = []
cited_author = []
citation_context = []
citation_class_label = []

# section_dict = {'introduction': 0, 'related work': 1, 'method': 2, 'experiments': 3, 'conclusion': 4}
label_dict = {'Background': 0, 'Extends': 1, 'Uses': 2, 'Motivation': 3, 'CompareOrContrast': 4, 'Future': 5}

with jsonlines.open(train_path, mode='r') as reader:
    for row in reader:
        # print(row['citing_paper_title'])
        # print(row['cited_paper_title'])
        # print(row.keys())
        # print(row.keys())
        # print(row['text'])
        # print(row['cited_author_ids'][0])
        # clean_text = row['cleaned_cite_text']
        # print(clean_text)
        # clean_text = re.sub('@@CITATION', '#AUTHOR_TAG', clean_text)
        # print(clean_text)
        # exit()
        # print(row['citing_paper_title'])
        # print(row['cited_paper_title'])

        # print(row['cited_author_ids'])
        # print(row['intent'])
        # print(row['cleaned_cite_text'])
        # print(row.keys())
        # print(type(row))
        # exit()

        core_id.append(row['citing_paper_id'])
        citing_title.append(row['citing_paper_title'])
        citing_author.append(row['citing_author_ids'])
        cited_title.append(row['cited_paper_title'])
        cited_author.append(row['cited_author_ids'][0])
        clean_text = row['cleaned_cite_text']
        text = re.sub('@@CITATION', '#AUTHOR_TAG', clean_text)
        citation_context.append(text)
        print(row['intent'])
        citation_class_label.append(label_dict[row['intent']])
# print(len(citation_class_label))
print(collections.Counter(citation_class_label))
exit()

# with jsonlines.open(dev_path, mode='r') as reader:
#     for row in reader:
#         core_id.append(row['citing_paper_id'])
#         citing_title.append(row['citing_paper_title'])
#         citing_author.append(row['citing_author_ids'])
#         cited_title.append(row['cited_paper_title'])
#         cited_author.append(row['cited_author_ids'])
#         # citation_context.append(row['text'])
#         clean_text = row['cleaned_cite_text']
#         text = re.sub('@@CITATION', '#AUTHOR_TAG', clean_text)
#         citation_context.append(text)
#         citation_class_label.append(label_dict[row['intent']])
# print(len(citation_class_label))
# exit()
dp = pd.DataFrame(columns=['unique_id', 'core_id', 'citing_title', 'citing_author',
                                         'cited_title', 'cited_author', 'citation_context', 'citation_class_label'])
# section_location = pd.DataFrame(columns=['unique_id', 'citation_class_label'])

# dp = pd.DataFrame(columns=['CGT', 'CDT', 'CC', 'label'])

# print(citation_class_label)

for i in range(len(citation_class_label)):
    print(i)
    dp.loc[i] = {
                # 'CGT': citing_title[i],
                 # 'CDT': cited_title[i],
                 # 'CC': citation_context[i],
                 # 'label': citation_class_label[i]}
                                'unique_id': i,
                                'core_id': core_id[i],
                                'citing_title': citing_title[i],

                                'citing_author': citing_author[i],
                                'cited_title': cited_title[i],
                                'cited_author': cited_author[i],
                                'citation_context': citation_context[i],
                                'citation_class_label': citation_class_label[i]}
    # print(dp.loc[i])
# print(dP)
# iitp_train = dp.loc[:int(dp.shape[0] * 0.9) - 1]
# iitp_vail = (dp.loc[int(dp.shape[0] * 0.9):]).reset_index(drop=True)
# iitp_train.to_csv('dataset/ireltrain.csv', sep=',', index=False, header=0)
# iitp_vail.to_csv('dataset/irelvail.csv', sep=',', index=False, header=0)
# dp.to_csv('dataset/iiptrain.csv', sep=',', index=False)
# dp.to_csv('dataset/iiptest.csv', sep=',', index=False)

