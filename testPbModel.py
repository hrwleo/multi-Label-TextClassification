import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
from model_train_gd_penalty import token_dict, OurTokenizer
import json


def load_model(model_dir):
    # 读取模型
    model = tf.contrib.predictor.from_saved_model(model_dir)
    return model


def printModel(model_dir):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)
        signature = meta_graph_def.signature_def
        print(signature)


def main():
    model_dir = "./model_pb"
    printModel(model_dir)
    model = load_model(model_dir)

    with open("label.json", "r", encoding="utf-8") as f:
        label_dict = json.loads(f.read())

    tokenizer = OurTokenizer(token_dict)
    text = "你好，能帮我详细介绍一下安元佳苑六区吗？"

    # 利用BERT进行tokenize
    maxlen = 64
    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)

    X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
    X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2

    feed_dict = {
        "segment_ids": [X2],  # [[]]
        "token_ids": [X1]  # [[]]
    }
    res = model(feed_dict)
    x_query_emb = res['output']
    one_hot = np.where(x_query_emb > 0.5, 1, 0)[0]
    print("预测标签: %s" % [label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]])


if __name__ == '__main__':
    main()
