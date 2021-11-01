# multi-Label-TextClassification
多标签文本分类

## 数据准备
> 多个标签以'，'逗号分割


## 模型结构

* 以下均适用albert-tiny预训练模型
> return_sequences：默认 False。在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。 False 返回单个， true 返回全部。

* 测试集：随机选取10w条样本做测试

|   Model  | Precision  | Recall | F1-score |
| :------------------------------------: | :------| :------| :------|
|albert + sigmoid| 0.9435 | 0.9287 | 0.9361 |
|albert + softmax+ CE + gd penalty| ___0.9846___ | 0.9403 | 0.9608 |
|albert + bilstm + sigmoid| 0.9635 | 0.9287 | 0.9446 |
|albert + bilstm + softmax + CE | 0.9673 | 0.9233 | 0.9432 |
|albert + bilstm + attention + sigmoid| 0.9627 | 0.9349 | 0.9478 |
|albert + textcnn + sigmoid| 0.9657 | 0.9378 | 0.9501 |
|albert + r-drop| ___0.9838___ | ___0.9622___ | ___0.9726___ |

### 1.albert-tiny预训练模型 + sigmoid

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 312)          0           model_2[1][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 55)           20345       lambda_1[0][0]                   
==================================================================================================
```

```
              precision    recall  f1-score   support
   micro avg     0.921     0.8987    0.9097    108248
   macro avg     0.9294    0.8770    0.9026    108248
weighted avg     0.9435    0.9287    0.9361    108248
```


### 2.albert-tiny预训练模型 + bilstm + sigmoid
```
BERT使用的是transformer，而transformer是基于self-attention的，也就是在计算的过程当中是弱化了位置信息的（仅靠position embedding来告诉模型输入token的位置信息），
而在任务当中位置信息是很有必要的，甚至方向信息也很有必要，所以需要用LSTM习得观测序列上的依赖关系


__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 312)          0           model_2[1][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1, 312)       0           lambda_1[0][0]                   
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 256)          451584      reshape_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 55)           14135       bidirectional_1[0][0]            
==================================================================================================
    
```

```
              precision    recall  f1-score   support
   micro avg     0.9641    0.9287    0.9461    108248
   macro avg     0.9494    0.8770    0.9097    108248
weighted avg     0.9635    0.9287    0.9446    108248
```


### 3.优化损失函数：albert-tiny预训练模型 + bilstm + (softmax + cross_entropy)

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 312)          0           model_2[1][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1, 312)       0           lambda_1[0][0]                   
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 256)          451584      reshape_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 55)           14135       bidirectional_1[0][0]            
==================================================================================================
```

```python
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
```


```
              precision    recall  f1-score   support
   micro avg     0.9687    0.9233    0.9455    108248
   macro avg     0.9549    0.8728    0.9088    108248
weighted avg     0.9673    0.9233    0.9432    108248
```

* loss优化-参考资料：
* https://kexue.fm/archives/7359 （softmax+交叉熵推广到多标签分类）
* https://zhuanlan.zhihu.com/p/153535799 （logsumexp， 真正意义上的softmax函数）
* https://zhuanlan.zhihu.com/p/375805722


### 4.albert-tiny预训练模型 + bilstm(return_sequences=True) + attention + sigmoid
```text
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 312)          0           model_2[1][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1, 312)       0           lambda_1[0][0]                   
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 1, 256)       451584      reshape_1[0][0]                  
__________________________________________________________________________________________________
attention_layer_1 (AttentionLay (None, 256)          66048       bidirectional_1[0][0]            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 55)           14135       attention_layer_1[0][0]          
==================================================================================================
```


```text
              precision    recall  f1-score   support
   micro avg     0.9633    0.9349    0.9489    108248
   macro avg     0.9444    0.8967    0.9184    108248
weighted avg     0.9627    0.9349    0.9478    108248
```

### 5.albert-tiny预训练模型 + textcnn + sigmoid
```text
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 312)          0           model_2[1][0]                    
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 1, 312)       0           lambda_1[0][0]                   
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 1, 312)       292344      reshape_1[0][0]                  
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 1, 312)       389688      reshape_1[0][0]                  
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 1, 312)       487032      reshape_1[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 1, 312)       0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 1, 312)       0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 1, 312)       0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1, 936)       0           max_pooling1d_1[0][0]            
                                                                 max_pooling1d_2[0][0]            
                                                                 max_pooling1d_3[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 936)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 936)          0           flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 55)           51535       dropout_1[0][0]                  
==================================================================================================
```

```
              precision    recall  f1-score   support
   micro avg     0.9659    0.9378    0.9516    108248
   macro avg     0.9514    0.8976    0.9212    108248
weighted avg     0.9657    0.9378    0.9501    108248
```


## 加入对抗训练提高模型性能
```text
    在一次竞赛经历中发现，使用了对抗训练，对模型的表现有所提升，因此在此做一个对比实验。
    深度学习中的对抗一般会有两个含义：一个是生成对抗网络，另一个是跟对抗攻击、对抗样本，主要是模型在小扰动下的稳健性
    （摘自：苏剑林. (Mar. 01, 2020). 《对抗训练浅谈：意义、方法和思考（附Keras实现） 》[Blog post]. Retrieved from https://kexue.fm/archives/7234）
    
    本次实验分两种对抗训练，一种是在模型的embeeding层添加扰动，另一种是在损失函数增加扰动，也叫梯度惩罚（通过损失函数对参数求梯度，代入GAN之父提出的快速梯度FGM，可得到扰动项）。
    
    在实现的过程中碰到的问题： keras自带的交叉熵函数不支持二阶梯度的计算，最后使用了softmax+交叉熵重写了损失函数
```

```text
    1. 在embedding层添加扰动
                    precision    recall  f1-score   support
    micro avg     0.9854    0.9403    0.9623    217879
    macro avg     0.9361    0.8760    0.9037    217879
 weighted avg     0.9846    0.9403    0.9608    217879
    
```

```python
def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数
```

```text
    2. 梯度惩罚
                    precision    recall  f1-score   support
    micro avg     0.9854    0.9403    0.9623    217879
    macro avg     0.9361    0.8760    0.9037    217879
 weighted avg     0.9846    0.9403    0.9608    217879
```

```python
# 参考原文作者的实现，把交叉熵改写为多标签分类的softmax + CE
def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embed-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
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
```


## R-Drop 正则化dropout
```text
它通过增加一个正则项，来强化模型对Dropout的鲁棒性，使得不同的Dropout下模型的输出基本一致，因此能降低这种不一致性，
促进“模型平均”与“权重平均”的相似性，从而使得简单关闭Dropout的效果等价于多Dropout模型融合的结果，提升模型最终性能
```

```python
数据输入要复制

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
```

```python
loss

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
```

```text
    2. r-drop
                    precision    recall  f1-score   support
   micro avg     0.9841    0.9622    0.9731    217879
   macro avg     0.9326    0.9059    0.9186    217879
weighted avg     0.9838    0.9622    0.9726    217879
```




## 多标签细粒度分类（扩展）

```text
    多标签分类的任务，是预测多个标签，每个类别只有0或1，可以直接在最后一层映射到lable数，用sigmoid+BCE做损失函数，把每个类别都当作二分类任务即可
    多标签的细粒度分类，比如细粒度情感分类，每个类别都是一个多分类任务，则需要对每个标签都做softmax
    
    举例：情感定义共6类（按顺序）：爱、乐、惊、怒、恐、哀,6类情感按固定顺序对应的情感值，情感值范围是[0, 1, 2, 3]，0-没有，1-弱，2-中，3-强
    在处理label的时候需要把标签转为6*4的二维数组，在计算损失的时候，对标签的第二维做softmax，损失函数可用CE
    
    数据集：https://www.datafountain.cn/competitions/518/datasets
    
    注：softmax + CE默认对最后一维计算
```
