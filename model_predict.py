# -*- coding: utf-8 -*-
# 模型预测脚本

import time
import json
import numpy as np

from model_train_rdrop import token_dict, OurTokenizer, multilabel_categorical_crossentropy_rdrop
from keras.models import load_model
from albert import get_custom_objects

maxlen = 64

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# 加载训练好的模型
model = load_model("albert_base_multi_label_ee.h5", custom_objects=merge_two_dicts(get_custom_objects(), {'multilabel_categorical_crossentropy_rdrop': multilabel_categorical_crossentropy_rdrop}))
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

s_time = time.time()
# 预测示例语句
text = "昨天18：30，陕西宁强县胡家坝镇向家沟村三组发生山体坍塌，5人被埋。当晚，3人被救出，其中1人在医院抢救无效死亡，2人在送医途中死亡。今天凌晨，另外2人被发现，已无生命迹象。"

# 利用BERT进行tokenize
text = text[:maxlen]
x1, x2 = tokenizer.encode(first=text)

X1 = x1 + [0] * (maxlen-len(x1)) if len(x1) < maxlen else x1
X2 = x2 + [0] * (maxlen-len(x2)) if len(x2) < maxlen else x2

X1 = [X1]
X2 = [X2]

# 模型预测并输出预测结果
prediction = model.predict([X1, X2])
one_hot = np.where(prediction > 0.5, 1, 0)[0]


print("原文: %s" % text)
print("预测标签: %s" % [label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]])
e_time = time.time()
print("cost time:", e_time-s_time)