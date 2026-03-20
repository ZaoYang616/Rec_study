import torch
import torch.nn as nn
import json

'''
ShareBottom_PEPNet (主模型)
│
├── 1. embedding_layer: MultiHotEmbeddingSum
│      └── embedding: nn.Embedding(num_embeddings=V, embedding_dim=16, padding_idx=0)
│
├── 2. epnet: EPNet (处理场景与物料)
│      └── gate_nu: GateNU 
│             └── net: nn.Sequential (Linear -> ReLU -> Linear -> Sigmoid)
│
├── 3. shared_bottom: nn.Sequential (点击塔特征提取)
│      ├── 0: nn.Linear(in=16, out=256)
│      ├── 1: nn.ReLU()
│      ├── 2: nn.Linear(in=256, out=128)
│      └── 3: nn.ReLU()
│
├── 4. click_head: nn.Sequential (点击塔输出)
│      ├── 0: nn.Linear(in=128, out=1)
│      └── 1: nn.Sigmoid()
│
├── 5. gate_nn: GateNU (桥接门控，控制流向转化塔的特征)
│      └── net: nn.Sequential (Linear -> ReLU -> Linear -> Sigmoid)
│
└── 6. cvr_tower: PPNetTower (转化塔)
       ├── layers: nn.ModuleList
       │      ├── 0: PPNetLayer (第一层个性化微调)
       │      │      ├── linear: nn.Linear(in=128, out=128)
       │      │      └── gate: GateNU (专属门控)
       │      │
       │      └── 1: PPNetLayer (第二层个性化微调)
       │             ├── linear: nn.Linear(in=128, out=64)
       │             └── gate: GateNU (专属门控)
       │
       └── final_proj: nn.Linear(in=64, out=1)
       (外接 cvr_act: nn.Sigmoid())
'''

class GateNU(nn.Module):  #门控组件
    """
    两层门控网络（NU）：用于为专家或特征动态生成缩放系数。
    结构：Linear(ReLU) -> Linear(Sigmoid) -> Gamma 缩放
    """
    def __init__(self, in_features, hidden_units, gamma=2.0): #接收三个参数：输入维度 in_features，隐藏层维度列表 hidden_units，以及放大系数 gamma（默认 2.0）。
        super(GateNU, self).__init__() #调用父类的初始化方法，注册模型参数。
        
        assert len(hidden_units) == 2, "GateNU needs exactly 2 hidden sizes" #断言防御机制，严格规定这个门控网络必须是两层（对应理论设计图）。
        self.gamma = gamma #将放大系数保存为类属性。

        self.net = nn.Sequential( #nn.Sequential：按顺序堆叠网络层的容器
            nn.Linear(in_features, hidden_units[0]), #第一层 nn.Linear 进行线性降维或升维，接着使用 nn.ReLU() 引入非线性。
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]), #第二层 nn.Linear 输出最终所需的维度，最后接 nn.Sigmoid() 将输出值严格压缩到 [0, 1] 的概率区间。
            nn.Sigmoid()
        )

    def forward(self, x): #forward 是前向传播函数。将输入 x 喂给刚才定义的 self.net，得到 [0, 1] 的向量，再乘以 self.gamma（即 2.0），最终返回取值范围在 [0, 2] 的调权向量。
        # 输出范围 (0, gamma)，通常 gamma=2
        return self.gamma * self.net(x)


class EPNet(nn.Module): #场景个性化网络
    #用于将场景特征（Domain）转化为门控，动态调整物品特征（Item）。
    """
    EPNet (Embedding Patch Network)
    使用场景 (Domain) 表征去动态 Scaling 物品 (Item) 表征
    """
    def __init__(self, domain_dim, emb_dim): #初始化函数接收场景维度 domain_dim 和物品维度 emb_dim。
        super(EPNet, self).__init__() #调用父类的初始化方法，注册模型参数。
        # Gate 的输入是 domain 和 emb 拼接
        self.gate_nu = GateNU(in_features=domain_dim + emb_dim, 
                              hidden_units=[emb_dim, emb_dim])  #实例化刚才定义的 GateNU。
                              #注意它的输入维度是 domain_dim + emb_dim（因为待会儿要把两者拼接），而输出维度是 emb_dim（为了和物品特征对齐相乘）。

    def forward(self, domain_emb, item_emb):
        # tf.stop_gradient(emb) 在 PyTorch 中对应 emb.detach()
        # 避免门控分支反向传播干扰原始特征学习
        concat_in = torch.cat([domain_emb, item_emb.detach()], dim=-1)  #在特征维度（最后一维，即 dim=-1）上进行拼接。 #<- detach 就是图里的 FF w/o BP
        #item_emb.detach()：极其关键的截断操作。将物品特征从当前计算图中分离出来。这样门控网络在反向传播时，梯度到了这里就会停止，不会流向原生物品 Embedding 造成污染。
        scale_factor = self.gate_nu(concat_in) #将拼接好的特征喂给门控网络，算出缩放因子 scale_factor。 <- 绿色的 Gate NN
        return scale_factor * item_emb  #scale_factor * item_emb：将缩放因子与原始物品特征进行逐元素相乘（Element-wise product），完成对物料的场景化微调。 <- 图里的 ⊗ 符号

class AFMLayer(nn.Module):
    """
    Attentional Factorization Machine (AFM) 交叉层
    用于自动学习不同特征域(Field)两两交叉后的重要性得分
    """
    def __init__(self, embed_dim, attention_factor=16, num_fields=4):
        super(AFMLayer, self).__init__()
        self.num_fields = num_fields
        
        # 注意力网络: W_a * ReLU(W_x * (e_i * e_j) + b_x)
        self.attention_W = nn.Linear(embed_dim, attention_factor)
        self.attention_h = nn.Linear(attention_factor, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emb_list):
        """
        emb_list: 包含 num_fields 个 Tensor 的列表，每个 Tensor 形状为 [Batch, EmbedDim]
        """
        row = []
        col = []
        # 遍历所有域，进行两两特征组合 (对于 4 个域，一共 6 种组合)
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(emb_list[i])
                col.append(emb_list[j])
                
        p = torch.stack(row, dim=1) # [Batch, 6, EmbedDim]
        q = torch.stack(col, dim=1) # [Batch, 6, EmbedDim]
        
        # 1. 计算两两点积交叉 (Hadamard Product)
        inner_product = p * q       # [Batch, 6, EmbedDim]
        
        # 2. 注意力得分计算
        attn_scores = self.attention_W(inner_product) # [Batch, 6, AttnFactor]
        attn_scores = torch.relu(attn_scores)
        attn_scores = self.attention_h(attn_scores)   # [Batch, 6, 1]
        
        # 3. Softmax 归一化注意力权重
        attn_weights = self.softmax(attn_scores)      # [Batch, 6, 1]
        
        # 4. 根据权重对交叉特征进行加权求和
        afm_out = torch.sum(attn_weights * inner_product, dim=1) # [Batch, EmbedDim]
        
        return afm_out

class DINAttentionLayer(nn.Module):
    """
    基于阿里 Deep Interest Network (DIN) 的目标注意力机制层
    """
    def __init__(self, embed_dim=32, hidden_units=[64, 32]):
        super(DINAttentionLayer, self).__init__()
        # 经典的 DIN 拼接法：输入 = [Query, Key, Query-Key, Query*Key] 
        # 维度是 embed_dim 的 4 倍
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, query, keys, mask=None):
        """
        query: 目标商品特征 (Target Item) [Batch, EmbedDim]
        keys: 历史行为序列 (Behavior Sequence) [Batch, SeqLen, EmbedDim]
        mask: 掩码序列，区分真实商品和 0 填充 [Batch, SeqLen, 1]
        """
        seq_len = keys.shape[1]
        
        # 1. 将 Query 在序列长度维度上进行复制，以便与每一个历史行为对齐
        # [Batch, EmbedDim] -> [Batch, 1, EmbedDim] -> [Batch, SeqLen, EmbedDim]
        queries = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 2. 构造注意力网络的输入
        attn_in = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        
        # 3. 计算注意力得分 (兴趣强度)
        attn_scores = self.mlp(attn_in) # 输出形状: [Batch, SeqLen, 1]
        
        # 4. 掩码处理 (非常重要！)
        # 把 padding 出来的假商品(ID=0)的得分强制归零，防止产生噪音干扰
        if mask is not None:
            attn_scores = attn_scores * mask
            
        # 5. DIN 的精髓：不使用 Softmax！保留用户真实的兴趣绝对强度。
        # 直接使用得分对历史行为序列进行加权求和
        # 矩阵乘法: [Batch, 1, SeqLen] @ [Batch, SeqLen, EmbedDim] -> [Batch, 1, EmbedDim]
        attn_out = torch.matmul(attn_scores.transpose(1, 2), keys)
        
        # 去掉多余的维度 -> [Batch, EmbedDim]
        return attn_out.squeeze(1)

class PPNetLayer(nn.Module): #带个性化门控的单层网络
    """
    PPNet 的单层：包含一个正常的 Linear，以及用 persona 算出的 Gate 做特征放缩
    """
    def __init__(self, in_features, out_features, persona_dim, dropout=0.0): 
        super(PPNetLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #初始化一个标准的线性层 nn.Linear、激活函数 nn.ReLU，以及用来防止过拟合的 nn.Dropout（以一定概率将神经元置 0）。

        # 用 persona 和本层的输入联合决定本层门控
        self.gate = GateNU(in_features=persona_dim + in_features,
                           hidden_units=[out_features, out_features])  #为这一层单独初始化一个 GateNU。输入是用户的 persona_dim 加上当前层的输入 in_features，输出和本层的 out_features 一致。

    def forward(self, x, persona): #前向传播时，接收上一层的输出 x 和全局不变的用户画像 persona。
        # 计算门控
        concat_in = torch.cat([persona, x.detach()], dim=-1) #同样使用 x.detach() 截断梯度，拼接后送入门控算出针对当前层特征的缩放系数 g
        g = self.gate(concat_in)
        
        # 线性变换后被门控缩放
        out = self.linear(x) #让 x 通过正常的线性映射得到 out
        out = g * out #用算出的个性化系数 g 乘以 out，实现参数的个性化控制
        return self.dropout(self.act(out)) #最后过 ReLU 激活并施加 Dropout，返回结果给下一层


class PPNetTower(nn.Module):
    """
    由多层 PPNetLayer 构成的任务塔（比如 CVR 塔）
    """
    def __init__(self, in_features, persona_dim, hidden_units=[128, 64], dropout=0.0):
        super(PPNetTower, self).__init__()
        self.layers = nn.ModuleList() #nn.ModuleList()：PyTorch 专用的列表容器。如果用普通的 Python list，里面包含的层不会被 PyTorch 识别，无法更新参数。必须用 ModuleList
        
        #动态构建多层网络。遍历 hidden_units 列表（例如 [128, 64]），不断实例化 PPNetLayer 并追加到列表中。更新 curr_in 为下一层的输入维度。
        curr_in = in_features
        for out_dim in hidden_units:
            self.layers.append(PPNetLayer(curr_in, out_dim, persona_dim, dropout))
            curr_in = out_dim
        # 最后一层输出 1 维 Logit
        
        #最后一层不需要门控，只是一个简单的线性映射，将最后一层的隐藏单元（例如 64 维）直接映射为 1 维标量（即未经 Sigmoid 激活的 Logit 分数）
        self.final_proj = nn.Linear(curr_in, 1)

    #遍历执行每一层，并将上一层输出作为下一层输入，用户 persona 贯穿始终。最后返回 1 维的得分。
    def forward(self, x, persona):
        for layer in self.layers:
            x = layer(x, persona)
        return self.final_proj(x)

class MultiHotEmbeddingSum(nn.Module):
    def __init__(self, vocab_json_path, embed_dim=16):
        super().__init__()
        with open(vocab_json_path, "r") as f:
            vocab_sizes = json.load(f)
            
        total_vocab_size = sum(vocab_sizes.values()) + max(vocab_sizes.values())
        self.embedding = nn.Embedding(num_embeddings=total_vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # 💡 新增：加入 LayerNorm 稳定数值分布
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x_idx):
        emb = self.embedding(x_idx)
        sum_emb = emb.sum(dim=1)
        # 💡 新增：对相加后的特征进行归一化
        return self.norm(sum_emb)

class ShareBottom_PEPNet(nn.Module):
    def __init__(self, vocab_json_path, embed_dim=32, 
                 shared_bottom_units=[256, 128], 
                 cvr_tower_units=[128, 64],
                 bottom_dropout=0.0,  
                 cvr_dropout=0.05):
        super(ShareBottom_PEPNet, self).__init__()
        
        self.embedding_layer = MultiHotEmbeddingSum(vocab_json_path, embed_dim)
        self.epnet = EPNet(domain_dim=embed_dim, emb_dim=embed_dim)
        self.afm = AFMLayer(embed_dim=embed_dim, attention_factor=16, num_fields=4)
        
        # 💡 新增: 实例化 DIN 注意力机制层
        self.din_attention = DINAttentionLayer(embed_dim=embed_dim)
        
        mlp_in_dim = embed_dim * 5
        
        self.shared_bottom = nn.Sequential(
            nn.Linear(mlp_in_dim, shared_bottom_units[0]),
            nn.ReLU(),
            nn.Dropout(bottom_dropout),
            nn.Linear(shared_bottom_units[0], shared_bottom_units[1]),
            nn.ReLU()
        )
        
        self.click_head = nn.Sequential(
            nn.Linear(shared_bottom_units[1], 1),
            nn.Sigmoid()
        )
        
        persona_dim = embed_dim * 2 
        self.gate_nn = GateNU(in_features=persona_dim, 
                              hidden_units=[shared_bottom_units[1], shared_bottom_units[1]])
        self.cvr_tower = PPNetTower(in_features=shared_bottom_units[1], 
                                    persona_dim=persona_dim, 
                                    hidden_units=cvr_tower_units,
                                    dropout=cvr_dropout)
        self.cvr_act = nn.Sigmoid()

    def forward(self, batch):
        # 1. 提取基础特征 (此时 item_emb 已经被暴力相加，没关系，作为 Target 是合理的)
        scene_emb = self.embedding_layer(batch['epnet_scene_idx'])
        item_emb = self.embedding_layer(batch['item_and_cross_idx'])
        profile_emb = self.embedding_layer(batch['user_profile_idx'])
        
        # 场景对物品进行门控缩放，生成最终的 Target Item
        patched_item_emb = self.epnet(scene_emb, item_emb)
        
        # 💡 核心改动开始 ==========================================
        # 提取用户历史行为的“原始序列”，不要直接调用 embedding_layer 导致它被暴力 Sum 掉
        # 提取出来的形状是 [Batch, 50, EmbedDim] (假设序列长度是50)
        behavior_seq = self.embedding_layer.embedding(batch['user_behavior_idx'])
        
        # 生成 Mask (判断序列里哪些位置是大于 0 的真实商品ID，哪些是 0 的占位符)
        # 形状转为 [Batch, 50, 1]
        seq_mask = (batch['user_behavior_idx'] > 0).float().unsqueeze(-1)
        
        # 真正的高级操作：让 Target Item 去检索用户行为序列，计算 Attention
        behavior_emb = self.din_attention(query=patched_item_emb, keys=behavior_seq, mask=seq_mask)
        
        # 为了防止加权求和后数值过大，仍然使用底层的 LayerNorm 稳定一下分布
        behavior_emb = self.embedding_layer.norm(behavior_emb)
        # 💡 核心改动结束 ==========================================
        
        # 下面的流程和以前一模一样，但此时的 behavior_emb 已经是拥有灵魂的动态向量了
        emb_list = [scene_emb, patched_item_emb, profile_emb, behavior_emb]
        afm_out = self.afm(emb_list)
        
        ctr_mlp_in = torch.cat([scene_emb, patched_item_emb, profile_emb, behavior_emb, afm_out], dim=-1)
        
        shared_hidden = self.shared_bottom(ctr_mlp_in)
        click_prob = self.click_head(shared_hidden).squeeze(-1) 
        
        persona_emb = torch.cat([profile_emb, behavior_emb], dim=-1)
        
        cvr_shared_hidden = self.gate_nn(persona_emb) * shared_hidden.detach() 
        cvr_logit = self.cvr_tower(cvr_shared_hidden, persona_emb)
        cvr_prob = self.cvr_act(cvr_logit).squeeze(-1)
        
        return click_prob, cvr_prob
