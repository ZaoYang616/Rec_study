# 推荐系统多任务学习 (MTL)：MMoE 模型深度解析

## 一、 核心痛点与架构演进

在多任务学习中，当不同任务的相关性较低或存在业务目标冲突时，传统的 Shared-Bottom（共享底层）架构会因为参数硬共享而引发严重的**负迁移（Negative Transfer）**问题。

为了解决这个问题，模型的底层架构经历了以下演进：
1. **OMoE (OneGate Mixture-of-Experts, 单门控混合专家)**：将底层共享的一个 Shared-Bottom 模块拆分成了多个 Expert（专家网络），最终输出为多个 Expert 的加权和。本质上是专家网络和全局门控的双层结构。
   * *局限性*：虽然提升了特征表征的多样性，但依然没有彻底解决多任务冲突问题，因为不同任务反向传播的梯度还是会通过同一个全局门控直接影响底层专家的学习。
2. **MMoE (Multi-gate Mixture-of-Experts, 多门控混合专家)**：为了进一步缓解多任务冲突，MMoE 为**每个任务配备专属的门控网络**，实现了门控从“全局共享”升级为“任务自适应”的方式。



---

## 二、 MMoE 的核心机制与数学表达

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/mmoe.png" width="400" />
</div>

MMoE 的精髓在于“专家会诊，各自导诊”。其前向传播可以通过以下四个公式清晰表达：

### 1. 专家网络输出 (Expert Network)
$$e_k = f_k(\mathbf{x})$$
* **$\mathbf{x}$**：底层特征输入。
* **$e_k$**：第 $k$ 个专家网络的输出。无论上面有几个任务，底层通常会构建 $K$ 个并行的专家网络，各自学习特定的特征模式。

### 2. 任务专属门控网络 (Gate Network)
$$g_t(\mathbf{x}) = \text{softmax}(\mathbf{W}_t \mathbf{x})$$
* **$g_t(\mathbf{x})$**：第 $t$ 个任务融合专家网络的门控向量。
* **直白理解**：门控网络会根据当前样本的特征，计算出任务 $t$ 应该给每个专家分配多少注意力权重（权重之和为 1）。例如在电商场景中，CTR 任务的门控可能会给“即时兴趣”专家分配高权重，而 CVR 任务的门控可能会侧重“消费能力”专家。

### 3. 专家加权融合 (Weighted Fusion)
$$\mathbf{h}_t = \sum_{k=1}^K g_{t,k} \cdot e_k$$
* **$\mathbf{h}_t$**：第 $t$ 个任务融合专家网络后的专属输出表示。
* **直白理解**：将所有专家的输出按照当前任务给定的权重进行加权求和，拼凑出最适合当前任务的特征向量。

### 4. 最终任务预测 (Task Prediction)
$$\hat{y}_t = f_t(\mathbf{h}_t)$$
* **$\hat{y}_t$**：第 $t$ 个任务的最终预测结果。把定制好的特征喂给任务 $t$ 自己的顶层塔即可。

---

## 三、 破局之道：MMoE 如何解决任务冲突？（梯度隔离）

当任务 $i$ 与任务 $j$ 在业务上发生冲突时，MMoE 的多门控机制会让两个任务学习到不同的专家权重分布，从而实现**梯度隔离**。

* **机制详解**：某个专家 $e_m$ 可能在任务 $i$ 的门控网络中获得很高的权重 $g_{i,m}$，而在任务 $j$ 的门控网络中获得很低的权重 $g_{j,m}$。
* **物理意义**：这样一来，专家 $e_m$ 的参数更新主要由任务 $i$ 的梯度决定，而任务 $j$ 的梯度对其影响很小，从而实现了梯度隔离。
* **结论**：不同任务通过选择不同的专家组合，可以各自学习到适合自己的特征表示，完美缓解了 Shared-Bottom 时代的“跷跷板”任务冲突现象。

---

## 四、 工业级代码实践拆解 (Keras/TensorFlow)

以下是 MMoE 架构构建的 Python 代码拆解：

```python
import tensorflow as tf

def build_mmoe_model(
    feature_columns,
    task_name_list,
    expert_nums=4,
    expert_dnn_units=[128, 64],
    gate_dnn_units=[128, 64],
    task_tower_dnn_units=[128, 64],
):
    # ---------------- 1. 基础特征处理 ----------------
    # 输入层：原始特征 -> Keras 输入
    input_layer_dict = build_input_layer(feature_columns)
    
    # 嵌入层：为各特征组创建嵌入表，得到组内嵌入向量
    embedding_table_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )
    
    # 合并嵌入：拼接为专家与门控的共同输入 (dnn_inputs)
    dnn_inputs = concat_group_embedding(embedding_table_dict, 'dnn')

    # ---------------- 2. 构建专家网络 (Experts) ----------------
    # 共享专家：多个并行 DNN（专家）供所有任务共享
    # 列表推导式生成 expert_nums 个独立的专家网络
    expert_outputs = [DNNs(expert_dnn_units, name=f"expert_{i}")(dnn_inputs) 
                      for i in range(expert_nums)]
    
    # 形状变换：将多个专家的输出堆叠起来，便于后续矩阵运算
    # 假设 Batch 大小为 B，专家数为 E，专家输出维度为 D
    # tf.stack 将列表变成形状为 [B, E, D] 的三维张量
    experts = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(expert_outputs)

    # ---------------- 3. 构建多门控与融合 (Gates & Fusion) ----------------
    task_features = []
    # 遍历每一个任务，为其生成专属门控权重并融合专家特征
    for idx, task_name in enumerate(task_name_list):
        # a. 门控网络隐藏层
        gate_hidden = DNNs(gate_dnn_units, name=f"task_{idx}_gate_mlp")(dnn_inputs)
        
        # b. 输出专家权重 (使用 Softmax 确保权重和为 1)
        # 输出形状 gate_weights: [B, E]
        gate_weights = tf.keras.layers.Dense(expert_nums, use_bias=False, 
                                             activation='softmax', 
                                             name=f"task_{idx}_gate_softmax")(gate_hidden) 
        
        # c. 加权融合 (神级操作 tf.einsum)
        # 'be' 代表 gate_weights 的维度 [Batch, Expert]
        # 'bed' 代表 experts 的维度 [Batch, Expert, Dim]
        # 'bd' 代表目标输出 task_mix 的维度 [Batch, Dim]
        # 这一步等价于 sum(gate_weights * experts)，计算出了当前任务专属的底层特征
        task_mix = tf.keras.layers.Lambda(
            lambda x: tf.einsum('be,bed->bd', x[0], x[1])
        )([gate_weights, experts])
        
        task_features.append(task_mix)

    # ---------------- 4. 构建任务独立塔 (Task Towers) ----------------
    task_outputs = []
    # 遍历任务，基于每个任务自己的融合特征 task_feat 构建独立塔
    for task_name, task_feat in zip(task_name_list, task_features):
        task_logit = DNNs(task_tower_dnn_units + [1])(task_feat)
        task_prob = PredictLayer(name=f"task_{task_name}")(task_logit)
        task_outputs.append(task_prob)

    # ---------------- 5. 封装最终模型 ----------------
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_outputs)
    return model
```