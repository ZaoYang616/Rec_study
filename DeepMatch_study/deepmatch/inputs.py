from itertools import chain

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, create_embedding_matrix, embedding_lookup, \
    get_dense_input, varlen_embedding_lookup, get_varlen_pooling_list, mergeDict


###！！！！非常无敌重要
##把数据（features）、配置（feature_columns）和模型层（Embedding Layers）全部连接起来，生成可以喂给 DNN 这里的输入张量。
#build_input_features: 这里打个洞，那里留个插座（定义 Input 占位符）。
#create_embedding_matrix: 准备家具，比如沙发、床（实例化 Embedding 矩阵）。
#input_from_feature_columns: 真正的布线和摆放 —— 把电线插到插座上，把沙发搬进客厅。它负责把输入数据流接到对应的矩阵上，算出来结果。
def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, embedding_matrix_dict=None):
    #这步常规操作，先把不同类型的特征配置分离开，因为它们的处理逻辑不一样。
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []


    #通常情况: 在 DSSM 代码中，我们是在外部先生成好了 embedding_matrix_dict（包含两塔所有的 embedding），然后传进来的
    #如果不传: 这里也会贴心地帮你现造一套。但不推荐这样做，因为如果双塔之间有共享 Embedding，分别造会导致共享失败。
    if embedding_matrix_dict is None:
        embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                        seq_mask_zero=seq_mask_zero)

    #处理普通离散特征（查表）
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    #Input: features (输入数据的 Tensor 字典), sparse_feature_columns (配置列表)。
    #Action: 遍历配置，拿出 Feature 名（比如 'user_id'），去 features 里拿数据 Tensor，再去 embedding_matrix_dict 里拿 Embedding 层，调用它。
    #Result: 得到的是一个字典，Key 是 Group Name（分组名，一般是 "default_group"），Value 是这时候查出来的 Embedding 向量（Tensor）。
    
    #处理数值特征（直接通过）
    dense_value_list = get_dense_input(features, feature_columns)
    #数值特征不需要查表（Embedding），直接拿数据本身就行了。或者可能经过一些简单的变换（如归一化）。

    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    #处理变长序列特征（高配查表 + 池化）
    # 先查表
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    #对序列里的每个 ID 都查表。
    #产出: 比如 hist_movie_id，本来输入是 (Batch, 50) 的整数，查完表变成了 (Batch, 50, EmbeddingDim) 的 3D 张量。

    #再池化
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    #把上面那个 (Batch, 50, Dim) 的东西，压缩成 (Batch, Dim)。
    #产出: 还是一个字典，Key 是 Group Name，Value 是池化后的 Embedding 向量（Tensor）。
    # 池化方式（mean/sum）取决于 VarLenSparseFeat 里的 combiner 参数。
    #关键点: 这里会自动处理 mask_zero。因为上面的查表结果携带了 Mask 信息（哪些是 0 补位的），这里 Pooling 的时候会自动忽略那些 0，只计算真实数据的平均值。

    #合并两种 Embedding 结果
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)

    #Flatten: 如果不需要分组（support_group=False），就把字典拍平，变成一个大列表
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))

    #Return:
    #group_embedding_dict: 所有的 Embedding 向量列表（比如 [User_vec, Gender_vec, Age_vec, Hist_vec]）
    #dense_value_list: 所有的数值特征列表。
    return group_embedding_dict, dense_value_list
