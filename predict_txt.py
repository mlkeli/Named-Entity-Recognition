import os
import json
import pandas as pd
import time
from tqdm import tqdm
from load_data import *
from model import *
from params import *
def merge_predictions(predictions):
    merged_predictions = []
    current_entity = None
    start_index = None
    for i, label in enumerate(predictions):
        if label.startswith("B-"):
            if current_entity is not None:
                merged_predictions.append((current_entity, start_index, i - 1))
            current_entity = label[2:]
            start_index = i
        elif label.startswith("I-"):
            if current_entity is not None and current_entity == label[2:]:
                continue
            else:
                if current_entity is not None:
                    merged_predictions.append((current_entity, start_index, i - 1))
                current_entity = None
                start_index = None
        elif label.startswith("E-"):
            if current_entity is not None and current_entity == label[2:]:
                merged_predictions.append((current_entity, start_index, i))
                current_entity = None
                start_index = None
            else:
                current_entity = None
                start_index = None
        else:
            current_entity = None
            start_index = None

    if current_entity is not None:
        merged_predictions.append((current_entity, start_index, len(predictions) - 1))

    return merged_predictions


def predict():
    train_filename = os.path.join('./data/datatrain6', 'train.txt')
    vocab = os.path.join(r'D:\nlp地址处理算法\LSTM_models\bert-base-chinese', 'vocab.txt')
    text = read_data1(vocab)
    train_text, train_label, A1 = read_data(train_filename)
    text6 = "甘肃省兰州市安宁区银滩雅苑第7幢104号房"
    # 读取txt文件
    with open('/kaggle/input/crm-shuju/CRM_.txt', 'r', encoding="utf-8") as file:
        lines = file.readlines()
        total = len(lines)
        start_time = time.time()

    BBB = []
    model = torch.load('./model/datatrain6/model_39.pth')  # 加载模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 初始化tokenizer
    with tqdm(total=total, ncols=80) as pbar:
        for i, line in enumerate(lines):
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (i + 1) * (total - i - 1)
            pbar.set_description(f"Progress: {i}/{total}")
            pbar.set_postfix({"Remaining Time": f"{remaining_time:.2f} seconds"})
            pbar.update(1)
            data = json.loads(line)
            text9 = data['text']
            inputs = tokenizer.encode(text9, return_tensors='pt')
            tokens = [text[idx] for idx in inputs.tolist()[0]]
            text11 = tokens[1:-1]
            text33 = ''.join(str(q) for q in text11)
            inputs = inputs.to(DEVICE)
            with torch.no_grad():  # 关闭梯度计算
                y_pre = model(inputs).reshape(-1)  # 进行预测
            _, id2label = build_label_2_index(train_label)
            label = [id2label[l] for l in y_pre[1:-1]]
            merged_predictions = merge_predictions(label)
            data = {}
            data["text"] = text33
            for entity, start_index, end_index in merged_predictions:
                text44 = text11[start_index:end_index + 1]
                text44 = ''.join(str(q) for q in text44)
                if entity in data:
                    data[entity] += text44
                else:
                    data[entity] = text44
            BBB.append(data)
        df = pd.DataFrame(BBB)
        df.to_csv("/kaggle/working/data1.csv", index=False)




if __name__ == '__main__':
    predict()