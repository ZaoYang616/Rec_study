import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq

class AliCCPParquetDataset(Dataset):
    """
    负责读取生成的 Parquet 文件。
    由于数据规模可以被加载到内存(约几百MB/GB级别), 直接用 Pandas 挂载以享受极高的随机读取速度。
    如果是巨大文件，可以替换为 pyarrow.parquet.ParquetFile 的按群组(RowGroup)游标读取。
    """
    def __init__(self, parquet_path):
        super().__init__()
        # 将 parqet 文件载入内存
        self.df = pd.read_parquet(parquet_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 返回单条样本的 Dictionary
        return {
            'click': torch.tensor(row['click'], dtype=torch.float32),
            'conversion': torch.tensor(row['conversion'], dtype=torch.float32),
            'sample_weight_conv': torch.tensor(row['sample_weight_conv'], dtype=torch.float32),
            
            # 对应的四个组件路径的数据
            'epnet_scene_idx': torch.tensor(row['epnet_scene_idx'], dtype=torch.long),
            'epnet_scene_val': torch.tensor(row['epnet_scene_val'], dtype=torch.float32),
            
            'user_profile_idx': torch.tensor(row['user_profile_idx'], dtype=torch.long),
            'user_profile_val': torch.tensor(row['user_profile_val'], dtype=torch.float32),
            
            'user_behavior_idx': torch.tensor(row['user_behavior_idx'], dtype=torch.long),
            'user_behavior_val': torch.tensor(row['user_behavior_val'], dtype=torch.float32),
            
            'item_and_cross_idx': torch.tensor(row['item_and_cross_idx'], dtype=torch.long),
            'item_and_cross_val': torch.tensor(row['item_and_cross_val'], dtype=torch.float32),
        }

def collate_fn_pad(batch):
    """
    配合 DataLoader 使用的数据整理补齐函数。
    它会一次性收到 N 行数据 (batch_size)，找出这批数据里的最长长度，然后自动 Padding 0。
    """
    # 抽取连续的 Label & Weights
    clicks = torch.stack([item['click'] for item in batch])
    conversions = torch.stack([item['conversion'] for item in batch])
    weights = torch.stack([item['sample_weight_conv'] for item in batch])
    
    def _pad_tensors(key, is_idx=True):
        # 提取当前 Batch 的某类变长数组
        tensors = [item[key] for item in batch]
        # idx 要 pad 为 0 (我们在前面制表时保留了0不被使用)；如果是特征权重 val 值 pad 为 0.0
        pad_val = 0 if is_idx else 0.0
        # batch_first=True 使得返回的 shape 是 (BatchSize, SeqLen)
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)
        return padded

    return {
        'click': clicks,
        'conversion': conversions,
        'sample_weight_conv': weights,
        
        'epnet_scene_idx': _pad_tensors('epnet_scene_idx', is_idx=True),
        'epnet_scene_val': _pad_tensors('epnet_scene_val', is_idx=False),
        
        'user_profile_idx': _pad_tensors('user_profile_idx', is_idx=True),
        'user_profile_val': _pad_tensors('user_profile_val', is_idx=False),
        
        'user_behavior_idx': _pad_tensors('user_behavior_idx', is_idx=True),
        'user_behavior_val': _pad_tensors('user_behavior_val', is_idx=False),
        
        'item_and_cross_idx': _pad_tensors('item_and_cross_idx', is_idx=True),
        'item_and_cross_val': _pad_tensors('item_and_cross_val', is_idx=False),
    }

def get_dataloader(parquet_path, batch_size=1024, shuffle=True, num_workers=4):
    """
    便捷函数获取封装好的 DataLoader 工具
    用法:
        train_loader = get_dataloader('train_model_input.parquet', batch_size=2048)
        for batch in train_loader:
             print(batch['user_profile_idx'].shape) # => [2048, 长度会自动对齐]
    """
    dataset = AliCCPParquetDataset(parquet_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True # 如果用 GPU 卡，可以加快到显存的速度
    )
    return loader