from load_data import *
from model import *
from params import *
import os
import json
from flask import Flask, request, jsonify
app = Flask(__name__)
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
            if current_entity is not None :
                continue
            else:
                if current_entity is not None:
                    merged_predictions.append((current_entity, start_index, i - 1))
                current_entity = None
                start_index = None
        elif label.startswith("E-"):
            if current_entity is not None:
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
    # 设置随机种子
    torch.manual_seed(0)
    # 设置设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_filename = os.path.join('./data', 'train.txt')
    vocab = os.path.join('./bert-base-chinese', 'vocab.txt')
    text = read_data1(vocab)
    train_text, train_label, A1 = read_data(train_filename)
    text6 = request.args.get('text')
    BBB = []
    model = torch.load('./model/model_39.pth', map_location="cpu")  # 加载模型
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')  # 初始化tokenizer
    inputs = tokenizer.encode(text6,add_special_tokens=True, truncation=True, return_tensors='pt')
    tokens = [text[idx] for idx in inputs.tolist()[0]]
    text11 = tokens[1:-1]
    text33 = ''.join(str(q) for q in text11)
    inputs = inputs.to(DEVICE)
    with torch.no_grad():  # 关闭梯度计算
        y_pre = model(inputs).reshape(-1)  # 进行预测
    _, id2label = build_label_2_index(train_label)
    label = [id2label[l] for l in y_pre[1:-1]]
    print(label)
    merged_predictions = merge_predictions(label)
    data = {}
    data["text"] = text33
    for entity, start_index, end_index in merged_predictions:
        text44 = text11[start_index:end_index + 1]
        text44 = ''.join(str(q) for q in text44)
        # if entity == "intersection":
        #     entity = "road"
        # elif  entity == "distance" or entity =="assist" or entity == "subpoi" or entity == "devzone":
        #     entity = "poi"
        if entity in data:
            data[entity] += text44
        else:
            data[entity] = text44
    BBB.append(data)
    response = {'predict': BBB}
    output = json.dumps(response, ensure_ascii=False)
    output = output.replace("#", "")  # 去除输出中的#号
    return output
@app.route('/predict', methods=['GET'])
def predict_route():
    output = predict()
    return output
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')