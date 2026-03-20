# 推荐系统多任务学习 (MTL)：Shared-Bottom 模型深度解析

## 一、 核心概念与基础架构

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/shared-bottom.png" width="400" />
</div>

在真实的搜广推场景中，业务目标往往是多元的（例如同时提升点击率 CTR 和转化率 CVR）。Shared-Bottom（共享底层）模型作为多目标建模的奠基性架构，采用了极其经典的 **“共享地基 + 独立塔楼”** 设计范式。

### 1. 架构组件
* **共享底层 (Shared Bottom)**：所有任务共用同一组特征转换层，负责学习跨任务的通用特征表示。
* **任务特定塔 (Task-Specific Towers)**：每个任务拥有独立的顶层网络，基于共享表示学习任务特定决策边界。

### 2. 数学表达
该架构的前向传播过程可用以下公式描述：
$$\hat{y}_t = f_t(\mathbf{W}_t \cdot g(\mathbf{W}_s \mathbf{x}))$$

* **$\mathbf{x}$**：原始输入特征。
* **$\mathbf{W}_s$**：共享层参数。不管上面有多少个任务，大家都在共同打磨这一个 $\mathbf{W}_s$。
* **$g(\cdot)$**：共享特征提取函数。所有任务共同更新这套参数。
* **$f_t(\cdot)$**：任务 $t$ 的预测函数。
* **$\mathbf{W}_t$**：任务 $t$ 的专属参数。

注意它的 **前提条件** ：这种设计哲学建立在“任务同质性假设”上，即认为不同的任务可以共享相同的底层特征空间。

---

## 二、 Shared-Bottom 的三大核心优势

Shared-Bottom 模型在效率与泛化之间实现了良好的平衡：

1. **极致的参数效率**：共享层占据了模型大部分的参数量，相比为每个任务单独建树，这显著降低了模型的总参数量，节省了线上推理算力。
2. **天然的正则化效应 (Regularization)**：
   * *什么是正则化？* 正则化是防止模型过拟合（死记硬背训练集数据）的约束手段。
   * *机制*：共享层如网同一个天然的正则化器，通过强制多个任务共用特征表示，底层网络无法只迎合某一个任务去“死记硬背”偏门特征，从而有效防止了单个任务出现过拟合的情况。
3. **高效的知识迁移**：当任务之间存在潜在的相关性时（例如视频的点击率与完播率），共享层能够学习到通用的模式，从而利用数据量大的任务帮助小样本任务提升泛化能力。

---

## 三、 致命缺陷：负迁移与“跷跷板”问题

当任务之间存在本质上的冲突时，该模型的硬共享机制会引发**负迁移 (Negative Transfer)** 问题。

* **业务现象**：例如在电商平台同时优化“点击率”与“客单价”时，低价商品可能会推动点击率的提升，但同时却抑制了客单价的增长。这种提升某一目标的性能往往需要以牺牲另一目标为代价的现象，被称为 **“跷跷板问题”**。
* **数学解释**：假设任务 $i$ 与任务 $j$ 的损失梯度分别为 $\nabla L_i$ 与 $\nabla L_j$。当 **$\nabla L_i \cdot \nabla L_j < 0$**（即梯度方向相反）时，共享层参数更新就会产生内在的冲突。模型在处理矛盾任务时呈现出“零和博弈”的特性。

---

## 四、 🚨 深度剖析：多目标任务中的样本选择偏差 (SSB)

在多目标任务中，**不同任务的输入特征（$X$）通常是完全共享的，但它们看待数据的样本空间和标签定义（$Y$）往往截然不同。**

以电商场景经典的“CTR（点击率） + CVR（转化率）”双目标预估为例：

1. **CTR 塔的视角**：
   * **样本空间**：整个大盘的**曝光数据**。
   * **标签**：曝光后点击为 1，未点击为 0。
2. **CVR 塔的视角**：
   * **样本空间**：仅限于**被点击过的数据**（未点击的商品无从谈起购买）。
   * **标签**：点击后购买为 1，未购买为 0。

### 带来的严峻挑战：
* **样本选择偏差 (Sample Selection Bias, SSB)**：CVR 塔在训练时，只能“看到”那些被点击过的商品数据。然而在线上真实推理时，CVR 塔需要对整个曝光大盘里的所有商品进行购买概率的预估。训练空间与推理空间发生严重断裂。
* **数据稀疏 (Data Sparsity, DS)**：相比于海量的曝光数据，点击数据通常只有前者的百分之几，这导致 CVR 任务面临极其严重的数据稀疏问题，极易过拟合。

*(注：为了解决上述硬共享冲突和样本偏差，工业界后续分别演化出了 MMoE 模型和 ESMM 空间混淆模型。)*

---

## 五、 Shared-Bottom 代码逻辑拆解

基于 TensorFlow/Keras 的核心构建逻辑如下：

```python
def build_shared_bottom_model(feature_columns, task_name_list, share_dnn_units=[128, 64], task_tower_dnn_units=[128, 64]):
    # 1. 输入层：将原始特征映射为 Keras 输入
    input_layer_dict = build_input_layer(feature_columns)
    
    # 2. 嵌入层：为各特征组创建嵌入表，得到组内嵌入向量
    embedding_table_dict = build_group_feature_embedding_table_dict(...)
    
    # 3. 合并嵌入：将多组嵌入拼接为共享 DNN 的长输入向量
    dnn_inputs = concat_group_embedding(embedding_table_dict, 'dnn')
    
    # 4. 【核心】共享底座 (Shared Bottom)
    # 所有特征强制经过这同一个 DNN 网络，提取通用特征
    shared_feature = DNNs(share_dnn_units)(dnn_inputs)
    
    # 5. 【核心】任务塔 (Task Towers)
    task_outputs = []
    # 遍历所有任务，在共享特征之上，为每个任务建立独立的 DNN 塔
    for task_name in task_name_list:
        task_logit = DNNs(task_tower_dnn_units + [1])(shared_feature) #在每一次循环中，都会独立地新建一个小型的 DNN 网络（任务塔）。重点是，这个塔的输入，正是刚才那个唯一的 shared_feature
        task_prob = PredictLayer(name=f"task_{task_name}")(task_logit) #每个塔最后输出自己的概率值。
        task_outputs.append(task_prob) #把所有塔的输出收集到一个列表中，用于后续计算各自的 Loss。
        
    # 6. 构建多任务模型：打包输入与多个任务塔的输出
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_outputs)
    return model 
 ```

代码总结：通过代码可以直观看出，shared_feature 是所有塔的唯一输入来源，这在代码结构上彻底锁死了“硬共享”的物理形态。