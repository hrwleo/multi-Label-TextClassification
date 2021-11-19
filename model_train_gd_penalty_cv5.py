# -*- coding: utf-8 -*-
import gc
import json
import codecs
import keras
import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras_bert import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import glob

import keras.backend as K
import tensorflow as tf

from albert import load_brightmart_albert_zh_checkpoint, get_custom_objects

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
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embed-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


# 构建模型
def create_cls_model(num_labels):
    albert_model = load_brightmart_albert_zh_checkpoint("./albert_tiny_489k", training=False)

    for layer in albert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = albert_model([x1_in, x2_in])
    cls_layer = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    # cls_layer = Reshape((-1, 312))(cls_layer)
    # bilstm = Bidirectional(LSTM(128, return_sequences=False), merge_mode='concat')(cls_layer)
    p = Dense(num_labels, activation=None)(cls_layer)  # 多分类
    # p = Lambda(lambda x: x ** 3)(p)

    model = Model([x1_in, x2_in], p)

    model.compile(
        loss=loss_with_gradient_penalty,
        # loss = 'binary_crossentropy',
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


from sklearn.model_selection import KFold


def run_cv(nfold, data, labels):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=1024).split(data)

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = [data[index] for index in train_fold], [data[index] for index in test_fold]
        model_i = create_cls_model(len(labels))

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)  # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5,
                                    patience=2)  # 当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint('albert_base_multi_label_ee.h5', monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')  # 保存最好的模型, 模型name固定，这样保存就会覆盖，不固定则会保存多个

        train_D = DataGenerator(X_train)
        valid_D = DataGenerator(X_valid)

        model_i.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=1,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        del model_i
        gc.collect()  # 清理内存
        K.clear_session()  # clear_session就是清除一个session


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


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
    # train_data = np.array(train_data)

    for i in range(test_df.shape[0]):
        label, content = test_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            for separate_label in label.split(","):
                if _ == separate_label:
                    label_id[j] = 1
        test_data.append((content, label_id))
    # test_data = np.array(test_data)

    # print(train_data[:10])
    print("finish data processing!")

    # # 模型训练
    # model = create_cls_model(len(labels))
    # train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)
    #
    # print("begin model training...")
    # model.fit_generator(
    #     train_D.__iter__(),
    #     steps_per_epoch=len(train_D),
    #     epochs=15,
    #     validation_data=test_D.__iter__(),
    #     validation_steps=len(test_D)
    # )

    run_cv(2, train_data, labels)

    print("finish model training!")

    # 模型保存
    # model.save('albert_base_multi_label_ee.h5')
    # model.load_weights('albert_base_multi_label_ee.h5')
    model = load_model("albert_base_multi_label_ee.h5",
                       custom_objects=merge_two_dicts(get_custom_objects(),
                                                      {'loss_with_gradient_penalty': loss_with_gradient_penalty}))
    export_savedmodel(model, './model_pb')

    print("Model saved!")

    result = model.evaluate_generator(test_D.__iter__(), steps=len(test_D))
    print("模型评估结果:", result)
