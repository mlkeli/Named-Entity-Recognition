import torch
from torch.utils.data import DataLoader, Dataset
from params import *


def read_data(filename):
    with open(filename, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []
    text = []
    labels = []
    max_length = 0  # 添加此行
    for data in all_data:
        if data == '':
            all_text.append(text)
            all_label.append(labels)
            max_length = max(max_length, len(text))
            text = []
            labels = []
        else:
            t, l = data.split('\t')
            text.append(t)
            labels.append(l)
    return all_text, all_label, max_length


def read_data1(filename):
    with open(filename, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    text = []
    for data in all_data:
        text.append(data)
    return text



def build_label_2_index(all_label):
    label_2_index = {'PAD': 0, 'UNK': 1}
    for labels in all_label:
        for label in labels:
            if label not in label_2_index:
                label_2_index[label] = len(label_2_index)
    return label_2_index, list(label_2_index)


class Data(Dataset):
    def __init__(self, all_text, all_label, tokenizer, label2index, max_length):
        self.all_text = all_text
        self.all_label = all_label
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label2index = label2index

    def __getitem__(self, item):
        text = self.all_text[item]
        labels = self.all_label[item][:self.max_length]

        # 需要对text编码，让bert可以接受
        text_index = self.tokenizer.encode(text,
                                           add_special_tokens=True,
                                           max_length=self.max_length + 2,
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt',
                                           )
        # 也需要将label进行编码
        # 那么我们需要构建一个函数来传入label2index
        # labels_index = [self.label2index.get(label, 1) for label in labels]
        # 上面那个就仅仅是转化，我们需要将label和text对齐
        labels_index = [0] + [self.label2index.get(label, 1) for label in labels] + [0] + [0] * (
                self.max_length - len(text))

        return text_index.squeeze(), torch.tensor(labels_index), len(text)

    def __len__(self):
        return len(self.all_text)