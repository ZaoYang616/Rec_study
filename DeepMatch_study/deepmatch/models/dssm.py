"""
Author:
    Zhe Wang, 734914022@qq.com
    Weichen Shen, weichenswc@163.com

Reference:
Huang P S , He X , Gao J , et al. Learning deep structured semantic models for web search using clickthrough data[C]// Acm International Conference on Conference on Information & Knowledge Management. ACM, 2013.
"""

from deepctr.feature_column import build_input_features, create_embedding_matrix
from deepctr.layers import PredictionLayer, DNN, combined_dnn_input
from tensorflow.python.keras.models import Model

from ..inputs import input_from_feature_columns
from ..layers.core import InBatchSoftmaxLayer
from ..utils import l2_normalize, inner_product


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, loss_type='softmax', temperature=0.05,
         sampler_config=None,
         seed=1024, ):
    """Instantiates the Deep Structured Semantic Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param loss_type: string. Loss type.
    :param temperature: float. Scaling factor.
    :param sampler_config: negative sample config.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    #1.准备embedding矩阵
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)
    '''
    user_feature_columns + item_feature_columns。
    把两边的特征合在一起初始化，如果两边有 Shared Embedding（比如都用了 embedding_name='movie_id'），这里会自动处理好共享关系，只创建一个矩阵。
    l2_reg_embedding：即对 Embedding 矩阵施加的 L2 正则化惩罚力度
        为什么要用它？
        防止过拟合：在推荐系统中，ID 类特征（如 User ID, Item ID）对应的 Embedding 矩阵通常非常巨大（参数量占整个模型的 90% 以上）。如果某些 ID 出现次数很少（稀疏），模型很容易“死记硬背”住这些 ID 的特征，导致泛化能力差。
        惩罚机制：L2 正则化会在 Loss 函数中增加一项 𝜆∑∣∣𝑤∣∣^2。它强迫 Embedding 向量里的数值尽可能小（接近于 0），不要有极端的突变值。
        效果：让训练出来的向量分布更平滑，模型更稳健。
        结果：这样一来，模型在训练时就不会过度依赖某些稀有 ID 的 Embedding，而是学到更通用、更平滑的特征表示，从而提升整体的泛化能力。
    seq_mask_zero=True：
        含义：告诉 Embedding 层，输入数据中的 0 是特殊的 Padding 占位符，不是真实的 ID。
        在 preprocess.py 里，我们用了 pad_sequences(..., value=0) 把用户的历史行为补齐到了 50 的长度。
        如果不开启 (False)：
            Embedding 层会把 0 当做一个正常的 ID 来处理，给它分配一个随机初始化的向量。
            这会引入噪音，因为 0 并不代表任何真实的物品。
        如果开启 (True)：
            屏蔽计算：模型在进行 Embedding 查表和后续计算时，会生成一个 Mask（掩码矩阵）。
            Embedding 层会忽略输入中的 0，不为它分配向量。
            这样，后续的池化层（Pooling Layer）在计算用户历史行为的整体向量时，就不会把这些 Padding 0 考虑进去，避免了噪音的干扰。
    '''
    '''
    本质: 一个 Python 字典 {'embedding_name': Keras_Embedding_Layer}。
    你可以把它想象成: “仓库里的货架”。
    内容:
        'user_id': 一个存了所有用户向量的巨型矩阵（Weights）。
        'movie_id': 一个存了所有电影向量的巨型矩阵。
        'gender': 一个存了性别向量的小矩阵。
    作用: 这是静态的参数集合。它不会随着每个 Batch 变化，它是等着被输入数据来“查”的表
    '''

    #2.构建User塔输入 
    user_features = build_input_features(user_feature_columns) #创建 Keras 的 Input 占位符。
    '''
    user_features（插座）

    本质: 一个 Python OrderedDict (有序字典) {'name': Keras_Input_Tensor}。
    你可以把它想象成: “墙上的插座面板”。
    内容:
    'user_id': 一个形状为 (batch, 1) 的占位符，等着插具体的 User ID。
    'hist_movie_id': 一个形状为 (batch, 50) 的占位符，等着插历史观看序列。
    作用: 这是模型的入口。它现在是空的，但在训练时，真实的数据会从这里灌进来。
    '''

    user_inputs_list = list(user_features.values()) #把字典的 Value（Input 层）取出来，放到一个 List 里，方便后续传给 Model。
    '''
    user_inputs_list（电源线束）
    本质: 一个 Python 列表 [Tensor, Tensor, ...]。
    你可以把它想象成: “从插座引出来的所有电源线”。
    内容: 也就是把 user_features 里的所有 Value（Input Tensor）取出来排成一排。
    [Tensor(user_id), Tensor(gender), Tensor(hist_movie_id), ...]
    作用: 这是为了最后定义 Model(inputs=...) 时方便。Keras 的 Model 需要你给它一个输入列表，告诉它：“这几个洞是给 User 塔用的”。
    '''
    
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    '''
     user_sparse_embedding_list（查出来的商品）
    本质: 一个 Python 列表 [Tensor, Tensor, ...]。
    你可以把它想象成: “刚从货架上取下来的商品（向量）”。
    内容:
    它拿着 user_features 里的 ID，去 embedding_matrix_dict 里查表。
    结果就是一堆向量。
    比如 User ID 对应的向量 (batch, 1, 32)，Gender 对应的向量 (batch, 1, 32)，还有历史行为 Pooling 后的向量 (batch, 1, 32)。
    作用: 此时，原始的整数 ID 已经变成了有意义的稠密向量。这些向量准备要被拼接起来了。

     user_dense_value_list（不需要查表的商品）
    本质: 一个 Python 列表 [Tensor, Tensor, ...]。
    你可以把它想象成: “自带的现金（数值）”。
    内容:
    如果你有数值型特征（比如 SparseFeat 和 VarLenSparseFeat 之外的 DenseFeat），比如“用户的年龄数值”、“用户的活跃度得分”。
    这些数据不需要查表，直接原样拿过来。
    作用: 它们会和上面的 Embedding 向量拼在一起。
    '''

    #input_from_feature_columns: 核心步骤。
    #根据 Input 里的 ID，去上面的 embedding_matrix_dict 里查表。
    #如果是序列特征（VarLen），在这里会自动做 Pooling（mean/sum）。
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list) #把 Sparse 和 Dense 特征拼接在一起，作为 DNN 的输入。
    '''
    数据源头 (x_train)
        ↓
    user_features (插座 Input) ----转成列表----> user_inputs_list (给 Model 用)
        ↓
    [查表操作: input_from_feature_columns] <--- 拿着 embedding_matrix_dict (货架)
        ↓
    user_sparse_embedding_list (查出来的向量)  +  user_dense_value_list (数值特征)
        ↓
    [拼接操作: combined_dnn_input]
        ↓
    user_dnn_input (拼成一条长向量)
        ↓
    DNN (神经网络)
    '''

    #3.构建Item塔输入
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)


    #4.通过DNN网络（多层全连接网络）
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation='linear', seed=seed)(user_dnn_input)
    #DNN: 全连接神经网络（MLP）。例如配置了 (128, 64, 32)，数据就会经过三层变换。
    #这就是“深度”学习的部分，负责进行特征交叉和非线性变换。
    user_dnn_out = l2_normalize(user_dnn_out)
    #把输出向量的模长强制变为 1。
    #作用：让后续的内积计算直接等价于余弦相似度

    #下同，但是物品塔只有两层（64，32）
    if len(item_dnn_hidden_units) > 0:
        item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                           dnn_use_bn, output_activation='linear', seed=seed)(item_dnn_input)
    else:
        item_dnn_out = item_dnn_input
    item_dnn_out = l2_normalize(item_dnn_out)

    #5.计算与输出LOSS
    #传统的二分类 (logistic)
    if loss_type == "logistic":
        score = inner_product(user_dnn_out, item_dnn_out, temperature)
        output = PredictionLayer("binary", False)(score)

    #各种 Softmax (softmax) <--- 本项目的模式
    elif loss_type == "softmax":
        output = InBatchSoftmaxLayer(sampler_config._asdict(), temperature)(
            [user_dnn_out, item_dnn_out, item_features[sampler_config.item_name]]) #item_name:在前面负采样设置部分。采样后的负样本 ID 需要去查 Embedding 表，这个名字必须和 item_feature_columns 里的名字对上。
    else:
        raise ValueError(' `loss_type` must be `logistic` or `softmax` ')

    #6.封装与返回
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    #为了后续方便取用，这里把 User 和 Item 的输入张量列表、Embedding 向量都挂到 model 里。
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)
    '''
    最后这几行非常贴心。
    训练时：我们要用完整的 model 来跑 fit。
    预测时：我们只想要 User 塔 (user_embedding) 去算用户向量，或者只想要 Item 塔 (item_embedding) 去算物品向量建库。
    DeepMatch 把这些中间层的 Tensor 挂载到 model 对象属性上，这样你在 Notebook 后半部分就能轻松地写出：
    user_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    '''


    return model
