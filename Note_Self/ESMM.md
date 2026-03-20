# 推荐系统多任务学习：ESMM 全空间多任务模型深度解析

在电商和推荐系统中，用户的行为有着极其严格的时序依赖：**曝光 (Impression) $\rightarrow$ 点击 (Click) $\rightarrow$ 转化 (Conversion)**。


<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/esmm_sample_bias.png" width="400" />
</div>


## 一、 业务背景与传统 CVR 模型的“死穴”

### 1. 北极星指标：CTR 与 CVR
* **CTR (点击率)**：`点击量 / 曝光量`。衡量商品的“外在吸引力”（如封面、标题）。
* **CVR (转化率)**：`转化量 / 点击量`。衡量商品的“内在匹配度”（如质量、价格）。
* **CTCVR (点击且转化率)**：`CTR × CVR = 转化量 / 曝光量`。这是平台最终追求的目标，即寻找既吸引人点、又能让人买单的优质商品。

### 2. 传统单独训练 CVR 模型的两大痛点
如果只拿“点击后”的数据去训练 CVR 模型，会面临两个极其致命的业务硬伤：
1. **样本选择偏差 (Sample Selection Bias, SSB)**：
   * **训练时**：模型只在“被点击过的样本”这一个小圈子里训练（见过的好东西多）。
   * **预测时**：模型却要在线上对大盘里“所有曝光的商品”（全量样本空间）进行转化率预估。
   * **后果**：训练空间和预测空间严重断裂，模型在线上“没见过世面”，泛化能力极差。
2. **数据稀疏性 (Data Sparsity)**：
   * 典型的电商大盘中，CTR 约 2%，CVR 约 0.5%，这意味着转化样本仅为曝光样本的万分之一。极其稀疏的正样本根本撑不起深度神经网络庞大的参数学习，极易过拟合。

---

## 二、 ESMM 的破局：全空间概率魔法

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/esmm.png" width="400" />
</div>

阿里提出的 ESMM 不改变底层的网络结构（如 Shared-Bottom 或 MMoE），而是通过改变**损失函数的计算空间**来解决偏差问题。


其核心源于一个极其简单的概率公式：
$$pCTCVR = pCTR \times pCVR$$

**ESMM 的架构设计**：
1. **输入层**：全量曝光样本（解决了 SSB 偏差）。
2. **CTR Tower**：预测点击率 $pCTR = f_{ctr}(\mathbf{h})$。
3. **CVR Tower**：预测转化率 $pCVR = f_{cvr}(\mathbf{h})$。
4. **乘法节点**：CVR Tower 预估出的 $pCVR$ **不直接计算 Loss**，而是与 $pCTR$ 相乘，得到 $pCTCVR$。拿着相乘后的联合概率，去全空间里算总 Loss。

---

## 三、 数学公式与损失函数设计

ESMM 的总 Loss 没有单独的 $\mathcal{L}_{CVR}$，全靠 CTR 和 CTCVR 来拉动：
$$\mathcal{L} = \mathcal{L}_{CTR} + \mathcal{L}_{CTCVR}$$

**1. CTR 损失 (标准的二分类交叉熵)**：
$$\mathcal{L}_{CTR} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i^{click} \log(pCTR_i) + (1 - y_i^{click}) \log(1 - pCTR_i) \right]$$

* 这是最标准的交叉熵。$N$ 代表全量曝光样本，点了 $y^{click}$ 就是 1，没点就是 0
* $N$ 代表全量曝光样本。

**2. CTCVR 损失 (偷天换日)**：
$$\mathcal{L}_{CTCVR} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i^{click} \cdot y_i^{conv} \log(pCTCVR_i) + (1 - y_i^{click} \cdot y_i^{conv}) \log(1 - pCTCVR_i) \right]$$
* 样本空间依然是 $N$（全量曝光）。
* 标签变成了复合标签 $y_i^{click} \cdot y_i^{conv}$：只有既点了又买了，标签才是 1，其他全是 0。名正言顺地在全量数据下进行训练。
* 通过这个约束，CTCVR 任务名正言顺地在全空间（无偏差）下进行了训练。

---

## 四、 ⭐️ 核心难点：没有 Loss，CVR 塔如何更新？

这是 ESMM 最精妙的数学把戏。既然没有 $\mathcal{L}_{CVR}$，CVR 塔的参数依靠**链式法则 (Chain Rule)** 从 CTCVR 节点“蹭”回梯度。

这意味着，CVR 塔是依靠 CTCVR 的误差，在整个大盘曝光数据上进行反向传播的。它从一开始就是在“全空间”的视角下学习的，彻底规避了训练集和测试集空间不一致的死穴。


前向传播有：$pCTCVR = pCTR \times pCVR$
根据链式法则，全局误差对 $pCVR$ 的偏导数为：
$$\frac{\partial \mathcal{L}_{CTCVR}}{\partial pCVR} = \frac{\partial \mathcal{L}_{CTCVR}}{\partial pCTCVR} \times \frac{\partial pCTCVR}{\partial pCVR} = \frac{\partial \mathcal{L}_{CTCVR}}{\partial pCTCVR} \times pCTR$$

**【业务直觉翻译：水管上的阀门】**
流进 CVR 塔的梯度大小，等于全局 CTCVR 的误差，乘以了 CTR 的预测概率 ($pCTR$)。**$pCTR$ 在这里充当了一个“阀门权重”**：
1. **当 $pCTR$ 极低（用户根本不想点）**：乘数接近 0，CVR 塔基本收不到梯度更新。模型认为“反正他不点，预测他买不买没意义，跳过”。
2. **当 $pCTR$ 很高（用户极大概率点）**：全局误差几乎 100% 流入 CVR 塔。模型认为“这人快进来了！CVR 塔赶紧用心算算他会不会买，狠狠更新参数”。

同时，共享层提取特征 $\mathbf{h}$ 承受了 CTR 和 CVR 两个任务的梯度拉扯，起到了极强的**正则化作用**，缓解了转化数据稀疏带来的过拟合问题。

---

## 五、 工业级代码实践拆解 (Keras/TensorFlow)

虽然底层是最简单的 Shared-Bottom，但通过 `Multiply` 层，极其优雅地实现了全空间约束。

```python
def build_esmm_model(
        feature_columns,
        task_tower_dnn_units=[128, 64],
        ):
    # 1) 输入与嵌入：构建输入层和分组嵌入，拼接为共享 DNN 输入
    input_layer_dict = build_input_layer(feature_columns)
    group_embedding_feature_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )
    dnn_inputs = concat_group_embedding(group_embedding_feature_dict, 'dnn')

    # 2) 双塔共享底座：同一输入分别走 CTR/CVR 塔，输出各自的 logit
    ctr_logit = DNNs(task_tower_dnn_units + [1], name="ctr_dnn")(dnn_inputs)
    cvr_logit = DNNs(task_tower_dnn_units + [1], name="cvr_dnn")(dnn_inputs)

    # 3) 概率与联乘：CTR 概率，CVR 概率；CTCVR = CTR × CVR
    ctr_prob = PredictLayer(name="ctr_output")(ctr_logit)
    cvr_prob = PredictLayer(name="cvr_output")(cvr_logit)
    ctcvr_prob = tf.keras.layers.Multiply()([ctr_prob, cvr_prob])

    # 4) 构建模型：输入为所有原始输入层，输出为 [CTR, CTCVR]
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=[ctr_prob, ctcvr_prob])
    return model
```

**实战进阶**：在真实的工业落地中，我们常常会将 ESMM 代码中的 Shared-Bottom 输入层 dnn_inputs 替换为 MMoE 或 PLE 提取的融合特征。这样就能达成“底层用 PLE 解决多任务特征冲突，上层用 ESMM 解决样本选择偏差”的终极排序模型组合！