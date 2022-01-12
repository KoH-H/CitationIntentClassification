import jsonlines
import pandas as pd
train_data = dict()
with open('dataset/acl/train.jsonl', 'r+', encoding='utf8') as f:
    for line in jsonlines.Reader(f):
        if 'citation_context' not in train_data:
            train_data['citation_context'] = [line['text']]
            train_data['citation_class_label'] = [line['intent']]
        else:
            context_list = train_data['citation_context']
            context_list.append(line['text'])
            label_list = train_data['citation_class_label']
            label_list.append(line['intent'])
            train_data['citation_context'] = context_list
            train_data['citation_class_label'] = label_list
train_data_df = pd.DataFrame(train_data)
print(train_data_df)