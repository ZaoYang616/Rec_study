import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("timestamp", inplace=True) # 按时间排序# 如果不排序，就会发生“用明天的行为预测昨天的行为”，导致特征穿越（Data Leakage）
    
    item_ids = data['movie_id'].unique() #为了后面做负采样时，知道候选池（Candidate Pool）有哪些。
    
    item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))
    # 构建一个 Python 字典：{movie_id: genres_id}。
    # 目的：为了快速查询某个电影ID对应的类型ID。
    # 当我们随机生成负样本（假电影）时，只知道它的ID，需要通过这个表查到它的类型，一起喂给模型。

    train_set = []
    test_set = []

    # 按用户分组，构建训练集和测试集
    for reviewerID, hist in tqdm(data.groupby('user_id')):  #把大表按用户拆分了
        ## reviewerID: 当前的用户ID。
        ## hist: 该用户的所有交互记录（DataFrame片段），因为前面排过序了，这里也是按时间有序的。

        pos_list = hist['movie_id'].tolist() #该用户看过的所有电影ID列表，顺序为 [电影1, 电影2, ..., 电影N]
        genres_list = hist['genres'].tolist() #对应的电影类型列表。
        rating_list = hist['rating'].tolist()#对应的评分列表。

        # 负采样
        if negsample > 0: #`只有当参数 negsample > 0 时才执行。是0这块会跳过。

            candidate_set = list(set(item_ids) - set(pos_list)) #候选负样本集合 = 全量物品集合 - 用户看过的正样本集合。
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True) #随机采样：一次性生成足够多的负样本。
            #数量是：正样本数 * 采样倍率。也就是给每个正样本都配好了相应数量的负样本。
        


        # 构建训练样本和测试样本
        #这是一个 滑动窗口（Sliding Window） 的过程。它模拟了用户随着时间推移，不断产生新行为的过程
        # 假如用户看了 [A, B, C, D]
            # i=1: 历史[A], 预测 B
            # i=2: 历史[A, B], 预测 C
            # i=3: 历史[A, B, C], 预测 D  <-- 最后一个
        for i in range(1, len(pos_list)):
            hist = pos_list[:i] ## 截取历史：从开始到当前时刻之前的所有电影作为“历史行为序列”。
            genres_hist = genres_list[:i] #对应的电影类型历史序列
            seq_len = min(i, seq_max_len) # 计算有效长度。如果历史太长，受限于模型的 seq_max_len，我们只能取最近的一部分。
            

            # -------------------------------------------------------------
            # 下面开始构造真正的样本。这里是 Leave-One-Out 测试集策略的核心。
            # -------------------------------------------------------------
            if i != len(pos_list) - 1:
                # 情况A：如果当前预测的目标 pos_list[i] 不是用户的最后一次行为。
                # 那么它属于【训练集】。
                train_set.append((
                    reviewerID,             # 用户ID
                    pos_list[i],            # 目标物品ID (即正样本)
                    1,                      # Label (1表示点击/看过)
                    #[::-1] 倒序操作：
                    #DeepMatch/DeepCTR 习惯将最近的行为放在前面。
                    hist[::-1][:seq_len],   # 历史序列。注意这里先 [::-1] 倒序，再截取最近的 seq_len 个。
                                            # 模型通常更关注最近的行为。
                    seq_len,                # 序列实际长度
                    genres_hist[::-1][:seq_len], # 历史类型序列
                    genres_list[i],         # 目标物品的类型
                    rating_list[i]          # 目标物品的评分
                ))

                # 接下来加入负样本 (negsample=0，这块不执行)
                for negi in range(negsample):
                    train_set.append((
                        reviewerID, 
                        neg_list[i * negsample + negi], # 取一个之前随机生成的负样本ID
                        0,                  # Label (0表示未点击)
                        hist[::-1][:seq_len], # 历史序列是一样的！同一个历史，既可以产生正样本，也可以产生负样本
                        seq_len,
                        genres_hist[::-1][:seq_len],
                        item_id_genres_map[neg_list[i * negsample + negi]] # 查表得到负样本的类型
                        # 评分没有，因为负样本是用户没看过的电影
                    ))
            
            # 情况B：如果 i == len(pos_list) - 1，说明这是用户的【最后一次行为】。
                # 我们把它留作【测试集】。
                # 这样可以测试模型根据先前的所有行为，能否预测出这最后一次行为。
            else:
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i],
                                 rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)
    # 打乱数据。这是训练神经网络的标准操作，防止模型记住数据的顺序（例如某个用户的数据都聚在一起），
    # 保证每个 Batch 的数据分布是独立同分布（IID）的。


    print(len(train_set[0]), len(test_set[0]))
    # 打印一条样本的长度，用于调试，确认格式对不对（比如是不是都是8个元素）。

    return train_set, test_set


def gen_data_set_sdm(data, seq_short_max_len=5, seq_prefer_max_len=50):
    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_short_len = min(i, seq_short_max_len)
            seq_prefer_len = min(max(i - seq_short_len, 0), seq_prefer_max_len)
            if i != len(pos_list) - 1:
                train_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))
            else:
                test_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


#它的核心作用是： 将前面 gen_data_set 生成的 Python 列表（List of Tuples）转换成 Keras/TensorFlow 模型能够直接喂入（Feed）的格式。
#Keras 的 model.fit 通常接收两个主要参数：x（输入特征）和 y（标签）。这个函数就是为了生成这两个东西。
def gen_model_input(train_set, user_profile, seq_max_len):
    #输入参数：
    #train_set: 上一步生成的列表，每个元素是一个巨大的元组：(user_id, item_id, label, history_seq, seq_len, history_genres, item_genres, rating)。
    #user_profile: 最开始处理好的 Pandas DataFrame，索引是 user_id，包含了用户的性别、年龄等静态属性。
    #seq_max_len: 序列的最大长度，用于 Padding（补全）。

    #解包数据
    #它把一个“行优先”的列表（每一行是一个样本），拆解成了“列优先”的多个数组（每一列是一个特征的所有样本数据）。
    train_uid = np.array([line[0] for line in train_set]) # 用户ID数组
    train_iid = np.array([line[1] for line in train_set]) # 物品ID数组
    train_label = np.array([line[2] for line in train_set]) # 标签数组（1或0）
    train_seq = [line[3] for line in train_set] #是一个列表，例如 [102, 55, 301, ...] 。
    # 历史行为序列列表 这里没有直接转 np.array，因为每个样本的序列长度可能不一样（有的看了3个，有的看了50个），NumPy 不支持锯齿状
    train_hist_len = np.array([line[4] for line in train_set]) # 历史序列实际长度数组 #这个特征对于变长序列模型（如 DIN, DIEN）非常重要，模型需要知道哪里是真实的记录，哪里是补零。
    #修改了train_seq_genres = np.array([line[5] for line in train_set])
    train_seq_genres = [line[5] for line in train_set] # 历史行为类型序列列表
    train_genres = np.array([line[6] for line in train_set]) # 目标物品的类型数组
######代码完全忽略了 line[7]（评分 rating），所以负样本没有rating也没关系，因为评分数据在这一步直接被丢弃了，根本没有入模。

    #序列 Padding
    #模型通常要求输入的序列长度一致，所以这里使用 Keras 的 pad_sequences 函数，把所有历史行为序列都补齐到相同长度 seq_max_len。
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post',
                                         value=0)
    #padding='post'：补零补在后面。例如 [1, 2] 变成 [1, 2, 0, 0, 0]。
    #有的模型喜欢 pre-padding（在前面补0），但 Transformer/RNN 类通常 post 居多， DeepMatch 里一般用 mask 处理 0，所以位置关系不大，但对应 train_hist_len 要一致。
    #truncating='post'：如果序列太长超过了 50，截断后面。
    #value=0：用 0 来填充空缺位。


    #构建模型输入字典
    #Key 必须和模型定义时的 Input 层名字一模一样。
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_genres": train_seq_genres_pad,
                         "hist_len": train_hist_len, "genres": train_genres}
    #user_id: 用户ID
    #movie_id: 目标物品离散特征
    #hist_movie_id: 变长历史序列特征
    #hist_genres: 变长历史类目序列特征
    #hist_len: 序列长度对应的数值特征
    #genres: 目标物品类目特征
    
    #添加用户静态特征
    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
        #user_profile.loc[...]：利用 Pandas 的强大索引功能。它会根据这个百万级的 ID 数组，去 user_profile 表里把对应的行全部取出来。这相当于 SQL 中的 Left Join 操作。
        #[key].values：取出对应列的值。
        #结果：假如 key 是 "gender"，这行代码会生成一个和训练集行数一样长的性别数组，直接塞进 train_model_input 字典里


    return train_model_input, train_label


def gen_model_input_sdm(train_set, user_profile, seq_short_max_len, seq_prefer_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    short_train_seq = [line[3] for line in train_set]
    prefer_train_seq = [line[4] for line in train_set]
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    #修改了    short_train_seq_genres = np.array([line[7] for line in train_set])
    #修改了    prefer_train_seq_genres = np.array([line[8] for line in train_set])
    short_train_seq_genres = [line[7] for line in train_set]
    prefer_train_seq_genres = [line[8] for line in train_set]

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_max_len, padding='post', truncating='post',
                                         value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_max_len, padding='post',
                                          truncating='post',
                                          value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_max_len, padding='post',
                                           truncating='post',
                                           value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_max_len, padding='post',
                                            truncating='post',
                                            value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "short_movie_id": train_short_item_pad,
                         "prefer_movie_id": train_prefer_item_pad,
                         "prefer_sess_length": train_prefer_len,
                         "short_sess_length": train_short_len, 'short_genres': train_short_genres_pad,
                         'prefer_genres': train_prefer_genres_pad}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
