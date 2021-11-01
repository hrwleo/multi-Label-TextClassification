# -*- coding: utf-8 -*-

import json
import codecs

import pandas as pd
import numpy as np
from keras_bert import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import tensorflow.keras.backend as K


import tensorflow as tf

from albert import load_brightmart_albert_zh_checkpoint

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

maxlen = 64
BATCH_SIZE = 128
dict_path = './albert_tiny_489k/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                # X1.append(x1)
                # X2.append(x2)
                # Y.append(y)
                for i in range(2):
                    X1.append(x1)
                    X2.append(x2)
                    Y.append(y)
                if len(X1) == self.batch_size * 2 or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def kld_binary(y_true, y_pred):
    y_true, y_pred = K.cast(y_true, tf.float32), K.cast(y_pred, tf.float32)
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    y_true_pos = K.clip(y_true, K.epsilon(), 1)
    y_pred_pos = K.clip(y_pred, K.epsilon(), 1)
    y_true_neg = K.clip(y_true_neg, K.epsilon(), 1)
    y_pred_neg = K.clip(y_pred_neg, K.epsilon(), 1)
    return K.sum(y_true_pos * K.log(y_true_pos / y_pred_pos) + y_true_neg * K.log(y_true_neg / y_pred_neg))


def total_loss(y_true, y_pred, alpha = 2):
    loss1 = K.mean(K.mean(K.binary_crossentropy(y_true, y_pred)))
    loss2 = K.mean(kld_binary(y_pred[::2], y_pred[1::2]) + kld_binary(y_pred[1::2], y_pred[::2]))
    return loss1 + alpha * loss2 / 4


# 构建模型
def create_cls_model(num_labels):
    albert_model = load_brightmart_albert_zh_checkpoint("./albert_tiny_489k", training=False, dropout_rate=0.3)

    for layer in albert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = albert_model([x1_in, x2_in])
    cls_layer = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类

    p = Dense(num_labels, activation='sigmoid')(cls_layer)  # 多分类

    model = Model([x1_in, x2_in], p)

    model.compile(
        loss=total_loss,
        optimizer=Adam(2e-5),
        metrics=['accuracy']
    )
    model.summary()

    return model


def export_savedmodel(model, export_path):
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'token_ids': model.input[0], 'segment_ids': model.input[1]}, outputs={'output': model.output})
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)  # 生成"savedmodel"协议缓冲区并保存变量和模型
    builder.add_meta_graph_and_variables(  # 将当前元图添加到savedmodel并保存变量
        sess=K.get_session(),  # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
        tags=[tf.saved_model.tag_constants.SERVING],  # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
        clear_devices=True,  # 清除设备信息
        signature_def_map={  # 签名定义映射
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  # 默认服务签名定义密钥
                model_signature  # 网络的输入输出策创建预测的签名
        })
    builder.save()
    print("save model pb success ...")


import glob
def input_fn(data_dir):
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


if __name__ == '__main__':

    _CSV_COLUMNS = [
        'label', 'content'
    ]

    # 数据处理, 读取训练集和测试集
    print("begin data processing...")
    train_df = input_fn("data")
    test_df = input_fn("data")

    select_labels = train_df["label"].unique()
    labels = []
    for label in select_labels:
        if "," not in label:
            if label not in labels:
                labels.append(label)
        else:
            for _ in label.split(","):
                if _ not in labels:
                    labels.append(_)
    with open("label.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(range(len(labels)), labels)), ensure_ascii=False, indent=2))

    train_data = []
    test_data = []
    for i in range(train_df.shape[0]):
        label, content = train_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            for separate_label in label.split(","):
                if _ == separate_label:
                    label_id[j] = 1
        train_data.append((content, label_id))

    for i in range(test_df.shape[0]):
        label, content = test_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            for separate_label in label.split(","):
                if _ == separate_label:
                    label_id[j] = 1
        test_data.append((content, label_id))

    # print(train_data[:10])
    print("finish data processing!")

    # 模型训练
    model = create_cls_model(len(labels))
    train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=15,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D)
    )

    print("finish model training!")

    # 模型保存
    model.save('albert_base_multi_label_ee.h5')
    # model.load_weights('albert_base_multi_label_ee.h5')
    export_savedmodel(model, './model_pb')

    print("Model saved!")

    result = model.evaluate_generator(test_D.__iter__(), steps=len(test_D))
    print("模型评估结果:", result)
