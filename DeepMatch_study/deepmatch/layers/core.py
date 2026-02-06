"""

Author:
    Weichen Shen,weichenswc@163.com

"""

import numpy as np
import tensorflow as tf
from deepctr.layers.utils import reduce_max, reduce_mean, reduce_sum, concat_func, div, softmax
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer


class PoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        self.sampler = self.sampler_config['sampler']
        self.item_count = self.sampler_config['item_count']

        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vocabulary_size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.vocabulary_size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        item_embeddings, user_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature
        if self.sampler == "inbatch":
            item_vec = tf.gather(item_embeddings, tf.squeeze(item_idx, axis=1))
            logits = tf.matmul(user_vec, item_vec, transpose_b=True)
            loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)

        else:
            num_sampled = self.sampler_config['num_sampled']
            if self.sampler == "frequency":
                sampled_values = tf.nn.fixed_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                       self.vocabulary_size,
                                                                       distortion=self.sampler_config['distortion'],
                                                                       unigrams=np.maximum(self.item_count, 1).tolist(),
                                                                       seed=None,
                                                                       name=None)
            elif self.sampler == "adaptive":
                sampled_values = tf.nn.learned_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                         self.vocabulary_size, seed=None, name=None)
            elif self.sampler == "uniform":
                try:
                    sampled_values = tf.nn.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                     self.vocabulary_size, seed=None, name=None)
                except AttributeError:
                    sampled_values = tf.random.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                         self.vocabulary_size, seed=None, name=None)
            else:
                raise ValueError(' `%s` sampler is not supported ' % self.sampler)

            loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,
                                              biases=self.zero_bias,
                                              labels=item_idx,
                                              inputs=user_vec,
                                              num_sampled=num_sampled,
                                              num_classes=self.vocabulary_size,
                                              sampled_values=sampled_values
                                              )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#这个类是 DeepMatch 实现 Sampled Softmax (负采样) 训练的核心黑科技，也是大规模推荐系统之所以能训练得动的关键
#它实现了一个非常巧妙的逻辑：“把 Batch 内的其他人当成你的负样本” (In-Batch Negative Sampling)。
class InBatchSoftmaxLayer(Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        self.item_count = self.sampler_config['item_count']
        #item_count: 每个物品被点击的次数（词频），用于纠正流行度偏差

        super(InBatchSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InBatchSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):

        # 解包输入，这里接收三个张量：
        # user_vec: [Batch_Size, Dim]  (每行是一个用户的最终向量)
        # item_vec: [Batch_Size, Dim]  (每行是该用户正反馈物品的最终向量)
        # item_idx: [Batch_Size, 1]    (每行是该物品的原始 ID，用于去查 LogQ 偏差)
        user_vec, item_vec, item_idx = inputs_with_item_idx
        #注意：这里传入的 item_vec 仅仅是当前 Batch 里那几个正样本物品的向量，没有传入全库几百万个物品的向量。

        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature  #温度系数调节
        #作用：缩放向量模长。temperature 越小，softmax 分布越尖锐（越自信）；temperature 越大，分布越平滑。这是一个调节模型区分度的超参数。

        #计算相似度矩阵
        # user_vec: [Batch, Dim]
        # item_vec: [Batch, Dim] (转置后变 [Dim, Batch])
        # matmul 结果 logits: [Batch, Batch]
        logits = tf.matmul(user_vec, item_vec, transpose_b=True)
        #这步在干嘛
        #它计算了 Batch 里 每一个用户 和 Batch 里 每一个物品 的点积（相似度）。
        #生成的矩阵 logits 是一个 N x N 的方阵。
        #矩阵的第 i 行第 j 列，表示 Batch 里第 i 个用户对 Batch 里第 j 个物品的打分（相似度）。
        #对角线上的元素 (logits[i][i])：第 i 个用户 VS 第 i 个物品。这是正样本（用户真的看了这个）。
        #非对角线元素 (logits[i][j], i!=j)：第 i 个用户 VS 第 j 个物品。这是负样本（把别人看的物品硬塞给你，当作你没看
#！！！！这就是 In-Batch 的精髓：不需要额外采样负样本，直接利用 Batch 内的其他样本作为负例，极大节省了计算量。

        #计算 Loss 并修正偏差 (LogQ Correction)
        loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
        base_config = super(InBatchSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!

        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = reduce_sum(keys * query, axis=-1, keep_dims=True)
        weight = tf.pow(weight, self.pow_p)  # [x,k_max,1]

        if len(inputs) == 3:
            k_user = inputs[2]
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)

        if self.pow_p >= 100:
            idx = tf.stack(
                [tf.range(tf.shape(keys)[0]), tf.squeeze(tf.argmax(weight, axis=1, output_type=tf.int32), axis=1)],
                axis=1)
            output = tf.gather_nd(keys, idx)
        else:
            weight = softmax(weight, dim=1, name="weight")
            output = tf.reduce_sum(keys * weight, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size)

    def get_config(self, ):
        config = {'k_max': self.k_max, 'pow_p': self.pow_p}
        base_config = super(LabelAwareAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        behavior_embedding = inputs[0]
        seq_len = inputs[1]
        batch_size = tf.shape(behavior_embedding)[0]

        mask = tf.reshape(tf.sequence_mask(seq_len, self.max_len, tf.float32), [-1, self.max_len, 1, 1])

        behavior_embedding_mapping = tf.tensordot(behavior_embedding, self.bilinear_mapping_matrix, axes=1)
        behavior_embedding_mapping = tf.expand_dims(behavior_embedding_mapping, axis=2)

        behavior_embdding_mapping_ = tf.stop_gradient(behavior_embedding_mapping)  # N,max_len,1,E
        try:
            routing_logits = tf.truncated_normal([batch_size, self.max_len, self.k_max, 1], stddev=self.init_std)
        except AttributeError:
            routing_logits = tf.compat.v1.truncated_normal([batch_size, self.max_len, self.k_max, 1],
                                                           stddev=self.init_std)
        routing_logits = tf.stop_gradient(routing_logits)

        k_user = None
        if len(inputs) == 3:
            k_user = inputs[2]
            interest_mask = tf.sequence_mask(k_user, self.k_max, tf.float32)
            interest_mask = tf.reshape(interest_mask, [batch_size, 1, self.k_max, 1])
            interest_mask = tf.tile(interest_mask, [1, self.max_len, 1, 1])

            interest_padding = tf.ones_like(interest_mask) * -2 ** 31
            interest_mask = tf.cast(interest_mask, tf.bool)

        for i in range(self.iteration_times):
            if k_user is not None:
                routing_logits = tf.where(interest_mask, routing_logits, interest_padding)
            try:
                weight = softmax(routing_logits, 2) * mask
            except TypeError:
                weight = tf.transpose(softmax(tf.transpose(routing_logits, [0, 1, 3, 2])),
                                      [0, 1, 3, 2]) * mask  # N,max_len,k_max,1
            if i < self.iteration_times - 1:
                Z = reduce_sum(tf.matmul(weight, behavior_embdding_mapping_), axis=1, keep_dims=True)  # N,1,k_max,E
                interest_capsules = squash(Z)
                delta_routing_logits = reduce_sum(
                    interest_capsules * behavior_embdding_mapping_,
                    axis=-1, keep_dims=True
                )
                routing_logits += delta_routing_logits
            else:
                Z = reduce_sum(tf.matmul(weight, behavior_embedding_mapping), axis=1, keep_dims=True)
                interest_capsules = squash(Z)

        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * inputs
    return vec_squashed

#"LogQ 修正" (LogQ Correction) 算法的实现，用于解决由于采样带来的数据分布偏差问题。
'''
1. 为什么要修正 (WHY)？
在 Negative Sampling（负采样）或者 In-Batch Sampling 中，我们并没有使用全部物品来计算 Softmax（分母太大算不动），而是只用了一小部分样本来近似。
问题：热门物品（Popular Items）在随机采样中被选为负样本的概率更高（因为它们出现的频率高）。
后果：如果不加以修正，模型会倾向于把热门物品的得分压得过低（因为模型经常要在负样本里“惩罚”它）。这会导致模型推荐出来的全是冷门物品，热门物品会被冤枉。
对策：我们在计算得分（logits）时，主动减去该物品流行度的对数 (logQ)。这就像给热门选手穿上“负重衣”，让比赛更公平。
直觉理解：如果你是一个热门物品，你得分高是理所当然的（即便没个性化推荐你也得分高）。既然你“自带流量”，那我要把你这部分自带的分扣掉，剩下的才是模型真正学到的“个性化匹配分”。
'''
def inbatch_softmax_cross_entropy_with_logits(logits, item_count, item_idx):
    #计算采样率Q
    # item_count: 各个物品的点击次数（频次统计列表）。
    # Q: 下面这行算出的是所有物品的 P(x) = count(x) / total_count
    #    然后 tf.gather 根据当前 Batch 里的 item_idx，把这几个物品的概率查出来。
    Q = tf.gather(tf.constant(item_count / np.sum(item_count), 'float32'),
                  tf.squeeze(item_idx, axis=1)) #Q 是一个形状为 (batch_size,) 的向量。比如 [0.1, 0.001, 0.05, ...], 分别对应 Batch 里第 1 个物品、第 2 个物品...在全局的出现概率。
    
    #计算 LogQ 并修正 Logits
    try:
        logQ = tf.reshape(tf.math.log(Q), (1, -1))  #取对数

        # 核心步骤！ logits -= logQ
        # logits 是 (Batch, Batch) 的矩阵。
        # logQ 是 (1, Batch) 的行向量。
        # 利用广播机制，logits 的每一行都会减去这同一个 logQ 向量。
        #数学公式：s(u,i)^=s(u,i) - log Q(i)
        #直觉理解：热门物品的 Q(i) 大，log Q(i) 也大，所以它们的得分 s(u,i) 会被大幅度扣减；冷门物品的 Q(i) 小，log Q(i) 也小，所以它们的得分 s(u,i) 扣减得少。
        #这样一来，热门物品就不会因为频次高而被模型过
        logits -= logQ  # subtract_log_q

        # 生成一个对角线为 1 的矩阵。
        # 例如 Batch=3:
        # [[1, 0, 0],
        #  [0, 1, 0],
        #  [0, 0, 1]]
        labels = tf.linalg.diag(tf.ones_like(logits[0]))
        #这告诉 Loss 函数：
        #对于第 0 个用户的 Logits 行，第 0 个位置（对自己拿的正样本）应该是最大的。
        #对于第 1 个用户的 Logits 行，第 1 个位置应该是最大的。
        #...以此类推。这就是 In-Batch 的逻辑：自己配对的是正样本，其他的都是负样本

    except AttributeError:
        logQ = tf.reshape(tf.log(Q), (1, -1))
        logits -= logQ  # subtract_log_q
        labels = tf.diag(tf.ones_like(logits[0]))


    #计算交叉熵损失
    #最后这一步就是标准的 Softmax Loss 计算。但因为输入的 logits 已经被 logQ 修正过了，所以这个 loss 是对采样偏差鲁棒的
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return loss


class EmbeddingIndex(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskUserEmbedding(Layer):

    def __init__(self, k_max, **kwargs):
        self.k_max = k_max
        super(MaskUserEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskUserEmbedding, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, training=None, **kwargs):
        user_embedding, interest_num = x
        if not training:
            interest_mask = tf.sequence_mask(interest_num, self.k_max, tf.float32)
            interest_mask = tf.reshape(interest_mask, [-1, self.k_max, 1])
            user_embedding *= interest_mask
        return user_embedding

    def get_config(self, ):
        config = {'k_max': self.k_max, }
        base_config = super(MaskUserEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
