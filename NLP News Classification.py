import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

NUM_CLASSES = 14  # 类别范围0-13
MAX_SEQ_LEN = 1024
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 数据读取与划分
train_df = pd.read_csv("train_set.csv", sep="\t")
test_df = pd.read_csv("test_a.csv", sep="\t")
print(f"原始训练集样本数：{len(train_df)}, 测试集样本数：{len(test_df)}")


def random_split(df, train_frac=0.8, val_frac=0.1, seed=123):
    assert train_frac + val_frac <= 1.0, "训练集和验证集比例之和不能超过1"
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


# 划分数据集（8:1:1）
train_df, val_df, test_df = random_split(train_df, 0.8, 0.1)
print(f"划分后 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")


# 数据集类定义
class NewsDataset(Dataset):
    def __init__(self, csv_path, max_seq_len=1024, pad_token_id=0, is_train=True,
                 stride=512, max_chunks=5, sep='\t'):
        self.df = pd.read_csv(csv_path, sep=sep)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.is_train = is_train
        self.stride = stride if stride is not None else max_seq_len // 2
        self.max_chunks = max_chunks
        self.processed_texts = self._preprocess_texts()

        if is_train:
            self.labels = self.df['label'].astype(int).tolist()  # 0-13类别

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.processed_texts[idx], dtype=torch.long)
        if self.is_train:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return text_tensor, label_tensor
        else:
            return text_tensor

    def _preprocess_texts(self):
        processed = []
        for text_str in self.df['text']:
            if pd.isna(text_str) or str(text_str).strip() == "":
                tokens = []
            else:
                tokens = [int(t) for t in str(text_str).split() if t.strip()]

            chunks = []
            start = 0
            while start < len(tokens) and len(chunks) < self.max_chunks:
                end = start + self.max_seq_len
                chunk = tokens[start:end]
                if len(chunk) < self.max_seq_len:
                    chunk += [self.pad_token_id] * (self.max_seq_len - len(chunk))
                chunks.append(chunk)
                start += self.stride

            if not chunks:
                chunks.append([self.pad_token_id] * self.max_seq_len)
            processed.append(chunks)
        return processed


# 自定义批次处理函数
def collate_fn(batch):
    if len(batch[0]) == 2:
        text_chunks_list, labels = zip(*batch)
        has_labels = True
    else:
        text_chunks_list = batch
        has_labels = False

    max_chunks = max(chunks.shape[0] for chunks in text_chunks_list)
    max_seq_len = text_chunks_list[0].shape[1]
    pad_token_id = 0

    padded_chunks = []
    for chunks in text_chunks_list:
        num_chunks = chunks.shape[0]
        if num_chunks < max_chunks:
            pad_chunks = torch.full((max_chunks - num_chunks, max_seq_len),
                                    pad_token_id, dtype=torch.long)
            chunks = torch.cat([chunks, pad_chunks], dim=0)
        padded_chunks.append(chunks)

    batch_chunks = torch.stack(padded_chunks)

    if has_labels:
        batch_labels = torch.stack(labels)
        return batch_chunks, batch_labels
    else:
        return batch_chunks


# 创建数据加载器
train_dataset = NewsDataset("train_1.csv", max_seq_len=MAX_SEQ_LEN, is_train=True)
val_dataset = NewsDataset("val_1.csv", max_seq_len=MAX_SEQ_LEN, is_train=True)
test_dataset = NewsDataset("test_1.csv", max_seq_len=MAX_SEQ_LEN, is_train=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, drop_last=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    num_workers=0, drop_last=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    num_workers=0, drop_last=False, collate_fn=collate_fn
)

# 验证数据加载器输出形状
for input_batch, target_batch in train_loader:
    print("输入批次形状:", input_batch.shape)
    print("标签批次形状:", target_batch.shape)
    print("标签示例:", target_batch[:5].tolist())
    break

