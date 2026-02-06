# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen, wcshenswc@163.com

"""

import json
import logging
import requests
from collections import namedtuple
from threading import Thread

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda


class NegativeSampler(
    namedtuple('NegativeSampler', ['sampler', 'num_sampled', 'item_name', 'item_count', 'distortion'])):
    """ NegativeSampler
    Args:
        sampler: sampler name,['inbatch', 'uniform', 'frequency' 'adaptive',] .
        num_sampled: negative samples number per one positive sample.   一般是Batch  Size -1
        item_name: pkey of item features . 采样后的负样本 ID 需要去查 Embedding 表，这个名字必须和 item_feature_columns 里的名字对上。
        item_count: global frequency of item .  就是费劲算出来的那个列表 [词频1, 词频2, ...]。用途: 计算 Log Probability，用于 Bias Correction（偏差修正）。
        公式: Logits = Logits - log(P(items))
        能防止模型因为某个物品出现在负样本里太频繁（因为它热门）而错误地压低它的分数。
        distortion: skew factor of the unigram probability distribution.  默认为 1.0，用来平滑 item_count 词频分布的。如果是 1.0，就是原始词频。如果是 0.75（Word2Vec 的经典值），会让长尾物品（冷门物品）被采样的概率稍微提升一点点。
    """
    #关于sampler 参数的说明：
#'uniform': 采用均匀分布进行负采样，所有负样本的选择概率相同。
#'frequency': 根据物品的出现频率进行负采样，出现频率高的物品被选中的概率更大。
#'adaptive': 自适应负采样，动态调整负样本的选择概率
#'inbatch' (最推荐，也是 Google 论文方案):
#原理: 不额外造假数据。直接把同一个 Batch 里其他用户看的正样本，当做我的负样本。
#优点: 极大地节省了 I/O 和 Embedding 查找的计算量。
#要求: item_count 必须传（用于纠正流行度偏差）。

    __slots__ = ()

    def __new__(cls, sampler, num_sampled, item_name, item_count=None, distortion=1.0, ):
        if sampler not in ['inbatch', 'uniform', 'frequency', 'adaptive']:
            raise ValueError(' `%s` sampler is not supported ' % sampler)
        if sampler in ['inbatch', 'frequency'] and item_count is None:
            raise ValueError(' `item_count` must not be `None` when using `inbatch` or `frequency` sampler')
        return super(NegativeSampler, cls).__new__(cls, sampler, num_sampled, item_name, item_count, distortion)

    # def __hash__(self):
    #     return self.sampler.__hash__()


def l2_normalize(x, axis=-1):
    return Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)
    #归一化
    #Cosine Similarity（余弦相似度）只关心方向，不关心长度。归一化后，内积（Inner Product）就直接等于余弦相似度了。这让训练更稳定。


def inner_product(x, y, temperature=1.0):
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])) / temperature)([x, y])
#计算两个向量的相似度（得分）
#temperature 参数用于调整得分的尺度（scale）。较高的 temperature 会使得得分更平滑，较低的 temperature 会使得得分更尖锐。

def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)
#y_true: 真实标签列表（用户实际看了哪些电影）。
#y_pred: 预测列表（模型推荐出来的排序列表，通常按分数从高到低已经排好序）。
#N=50: 截断值（Top K），只考察推荐列表的前 50 个。
#y_pred[:N]: 截取预测列表的前 N 个元素。
#set(...) & set(...): 利用 Python 的集合运算求交集。也就是看“预测的前 N 个”和“用户实际看的”里面，有几个是重合的（命中了几个）。
#* 1.0 / len(y_true): 除以真实标签的总数，得到召回率。
#在你的 Notebook (colab_MovieLen1M_DSSM_InBatchSoftmax.ipynb) 中，测试集是把用户的最后那一次点击拿出来做测试，所以对于每个用户：
    #y_true 的长度固定是 1。
    #recall_N 的计算结果要么是 0（没猜中），要么是 1.0（猜中了）。


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)
    #解释:
    #模型的输出 y_pred 其实就是那个 Layer 算出来的 Loss 值本身。
    #所以这里的 compile 里的 loss 函数只需要做一个简单的“搬运工”，直接把 y_pred（模型输出的损失）取个平均值返回给 Keras 框架就行了。
    #Keras 框架以为它在算 Loss，其实它只是在读 InBatchSoftmaxLayer 算好的结果。


def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepmatch/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning(
                        '\nDeepMatch version {0} detected. Your version is {1}.\nUse `pip install -U deepmatch` to upgrade.Changelog: https://github.com/shenweichen/DeepMatch/releases/tag/v{0}'.format(
                            latest_version, version))
        except:
            print("Please check the latest version manually on https://pypi.org/project/deepmatch/#history")
            return

    Thread(target=check, args=(version,)).start()


'''
NegativeSampler 如何工作?
NegativeSampler 本身（正如你在 utils.py 看到的）只是一个存数据的容器，它自己不干活。真正干活的是 DSSM 模型内部的代码。
创建了模型：model = DSSM(..., sampler_config=sampler_config)
调用 DSSM 函数时，DeepMatch 内部会读取这张纸条，并构建相应的 TensorFlow 计算图

# 伪代码逻辑，模拟 DeepMatch 内部实现
if sampler_config.sampler == 'inbatch':
    # 1. 获取 Batch 内所有物品的 Embedding
    # 在 In-Batch 模式下，Batch 里所有的正样本物品，都会被拿来当做"候选池"
    # 所以 item_embedding 本身就是候选池
    
    # 2. 计算相似度矩阵 (Batch_Size x Batch_Size)
    # 每一行(User) 都会和 每一列(Item) 计算内积
    # Result[i][j] 代表 User[i] 对 Item[j] 的兴趣分
    scores = tf.matmul(user_embedding, tf.transpose(item_embedding)) 
    
    # 3. 偏差校正 (Bias Correction) - 关键一步！
    # 取出之前传进去的 item_count，计算 log(P)
    # P(i) = item_count[i] / total_count
    
    # 获取当前 Batch 里所有物品对应的概率 bias
    # 这是一个查表操作，根据 Item ID 查刚才算好的概率
    item_bias = tf.gather(log_probability, item_ids) 
    
    # 从分数里减去偏差
    # Logits = DotProduct - log(Prob)
    # 只有减去了这个，才能放心地把热门物品当负样本
    scores = scores - item_bias 
    
    # 4. 计算 Softmax Loss
    # 目标：对于 User[i]，希望 scores[i][i] (正样本得分) 最大
    # 其他 scores[i][j] (负样本得分) 最小
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=range(Batch_Size), logits=scores)

'''