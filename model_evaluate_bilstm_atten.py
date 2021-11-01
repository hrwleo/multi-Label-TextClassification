# -*- coding: utf-8 -*-

# 模型评估脚本,利用hamming_loss作为多标签分类的评估指标，该值越小模型效果越好
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from albert import get_custom_objects
from sklearn.metrics import hamming_loss, classification_report
import os
from train_model_bilstm_atten import token_dict, OurTokenizer, AttentionLayer

maxlen = 64

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


# 加载训练好的模型
model = load_model("albert_base_multi_label_ee.h5",
                   custom_objects=merge_two_dicts(get_custom_objects(), {'AttentionLayer': AttentionLayer}))
tokenizer = OurTokenizer(token_dict)

with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


# 对单句话进行预测
def predict_single_text(text):
    # 利用BERT进行tokenize
    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)
    X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
    X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2

    X1 = [X1]
    X2 = [X2]

    # 模型预测并输出预测结果
    prediction = model.predict([X1, X2])
    one_hot = np.where(prediction > 0.5, 1, 0)[0]
    return one_hot, "|".join([label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]])


import glob


def input_fn(data_dir):
    _CSV_COLUMNS = [
        'label', 'content'
    ]
    read_file = glob.glob(os.path.join(data_dir, 'part*'))  # 读取文件夹中所有part-* 文件
    df = None
    for i, path in enumerate(read_file):
        try:
            data_ = pd.read_csv(path, header=None, names=_CSV_COLUMNS, sep='\t', error_bad_lines=False).fillna(value="")
            if df is None:
                df = data_
            else:
                df = pd.concat([df, data_], ignore_index=True)
        except:
            continue
    return df


# 模型评估
def evaluate():
    test_df = input_fn("stefan_multi_label_test_data")
    true_y_list, pred_y_list = [], []
    true_label_list, pred_label_list = [], []
    content_list = []
    common_cnt = 0
    for i in range(test_df.shape[0]):
        # print("predict %d samples" % (i + 1))
        true_label, content = test_df.iloc[i, :]
        true_y = [0] * len(label_dict.keys())
        for key, value in label_dict.items():
            if value in true_label:
                true_y[int(key)] = 1

        pred_y, pred_label = predict_single_text(content)
        if set(true_label.split("|")) == set(pred_label.split("|")):
            common_cnt += 1
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)
        true_label_list.append(true_label)
        pred_label_list.append(pred_label)
        content_list.append(content)

    # F1值
    print(classification_report(true_y_list, pred_y_list, digits=4))
    return true_label_list, pred_label_list, hamming_loss(true_y_list, pred_y_list), common_cnt / len(true_y_list), content_list


true_labels, pred_labels, h_loss, accuracy, content_list = evaluate()
df = pd.DataFrame({"content": content_list, "y_true": true_labels, "y_pred": pred_labels})
df.to_csv("albert_tiny_pred_result.csv")

print("accuracy: ", accuracy)
print("hamming loss: ", h_loss)
