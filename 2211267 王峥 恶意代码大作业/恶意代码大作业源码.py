# malware_detection_optimized.py

import jsonlines
import numpy as np
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import os
from tqdm import tqdm


# =========================
# 数据处理类
# =========================

def wash(names):
    if not isinstance(names, list):
        return []
    cleaned = []
    for name in names:
        tokens = ''.join([c if c.isalnum() else ' ' for c in name]).lower().split()
        cleaned.extend(tokens)
    return cleaned


class ReadData:
    def __init__(self, data_paths, num_samples=None):
        """
        data_paths: list of file paths to read data from
        num_samples: total number of samples to read (optional)
        """
        self.data_paths = data_paths
        self.num = num_samples
        self.data_list = []

    def split_data(self):
        for path in self.data_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    self.data_list.append(item)
                    if self.num and len(self.data_list) >= self.num:
                        break
            if self.num and len(self.data_list) >= self.num:
                break

        datas = []
        labels = []
        for data in self.data_list:
            if data['label'] >= 0:
                datas.append(data)
                labels.append(data['label'])
        return datas, labels


class FirstFeature:
    def __init__(self, texts):
        self.texts = texts

    def byteentropy(self):
        entropy = []
        for text in self.texts:
            entropy.append(text.get('byteentropy', [0] * 256))
        return np.array(entropy)


class SecFeature:
    def __init__(self, texts):
        self.texts = texts
        self.imports_name = self.imports()

    def imports(self):
        imports_name = []
        for text in self.texts:
            imports_name.append(wash(text.get('imports', [])))
        return imports_name

    def custom_hash(self, input_str):
        hash_object = hashlib.sha256(input_str.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)
        return hash_int

    def hash_to_256(self):
        name_to_256 = []
        for imp_name in self.imports_name:
            every_num = []
            for single_name in imp_name:
                every_num.append(self.custom_hash(single_name) % 256)
            name_to_256.append(every_num)

        all_hash256 = []
        for nums in name_to_256:
            hash256 = [0] * 256
            for num in nums:
                hash256[num] += 1
            all_hash256.append(hash256)
        return np.array(all_hash256)


class ThiFeature:
    def __init__(self, texts):
        self.texts = texts

    def get_histogram(self):
        histogram = []
        for text in self.texts:
            histogram.append(text.get('histogram', [0] * 256))
        return np.array(histogram)


class FourFeature:
    def __init__(self, texts):
        self.texts = texts
        self.header_name = self.header()

    def header(self):
        header_name = []
        for text in self.texts:
            header_name.append(wash(text.get('header', [])))
        return header_name

    def custom_hash(self, input_str):
        hash_object = hashlib.sha256(input_str.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)
        return hash_int

    def hash_to_256(self):
        name_to_256 = []
        for hdr_name in self.header_name:
            every_num = []
            for single_name in hdr_name:
                every_num.append(self.custom_hash(single_name) % 256)
            name_to_256.append(every_num)

        all_hash256 = []
        for nums in name_to_256:
            hash256 = [0] * 256
            for num in nums:
                hash256[num] += 1
            all_hash256.append(hash256)
        return np.array(all_hash256)


# =========================
# 数据集类
# =========================

class MalwareDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =========================
# 神经网络模型
# =========================

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.batch = nn.BatchNorm1d(output_size)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        if input_size != output_size:
            self.residual = nn.Linear(input_size, output_size)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.linear(x)
        out = self.batch(out)
        out = self.prelu(out)
        out = self.dropout(out)
        out += residual
        return out


class MalwareModel(nn.Module):
    def __init__(self, input_size=1024, hidden_sizes=[1024, 512, 256], num_classes=2, dropout=0.5):
        super(MalwareModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, hidden_size, dropout))
            prev_size = hidden_size
        self.layers = nn.Sequential(*layers)
        self.clas = nn.Linear(prev_size, num_classes)
        self.soft = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.pre = None

    def forward(self, x, label=None):
        x = torch.log10(1 + x)
        x = self.layers(x)
        x = self.clas(x)
        p = self.soft(x)
        self.pre = torch.argmax(p, dim=-1).detach().cpu().numpy().tolist()

        if label is not None:
            loss = self.loss_fn(p, label)
            return loss, p
        return p


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.uniform_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


# =========================
# 训练函数
# =========================

def train_model(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="训练中", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        loss, outputs = model(inputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1


# =========================
# 评估函数
# =========================

def evaluate_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="评估中", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            loss, outputs = model(inputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1


# =========================
# 主函数
# =========================

def main():
    # 配置
    TRAIN_DATA_PATHS = [
        'D:/ember_dataset_2018_2/ember2018/train_features_1.jsonl',
        'D:/ember_dataset_2018_2/ember2018/train_features_2.jsonl',
        'D:/ember_dataset_2018_2/ember2018/train_features_3.jsonl',
        'D:/ember_dataset_2018_2/ember2018/train_features_4.jsonl',
        'D:/ember_dataset_2018_2/ember2018/train_features_5.jsonl'
    ]  # 更新为您的训练数据路径列表
    TEST_DATA_PATH = 'D:/ember_dataset_2018_2/ember2018/test_features.jsonl'  # 更新为您的测试数据路径

    NUM_SAMPLES = None  # 如果不限制样本数，可设为 None
    BATCH_SIZE = 256
    NUM_EPOCHS = 50  # 增加训练轮次
    LEARNING_RATE = 0.001
    VALID_SPLIT = 0.1
    RANDOM_SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================
    # 数据加载与预处理
    # =========================
    print("加载和处理训练数据中...")
    reader = ReadData(TRAIN_DATA_PATHS, NUM_SAMPLES)
    texts, labels = reader.split_data()

    # 特征提取
    entropy = FirstFeature(texts).byteentropy()
    hash256 = SecFeature(texts).hash_to_256()
    histogram = ThiFeature(texts).get_histogram()
    header_hash = FourFeature(texts).hash_to_256()

    # 转换为张量
    entropy = torch.tensor(entropy, dtype=torch.float32)
    hash256 = torch.tensor(hash256, dtype=torch.float32)
    histogram = torch.tensor(histogram, dtype=torch.float32)
    header_hash = torch.tensor(header_hash, dtype=torch.float32)

    # 拼接特征
    features = torch.cat([entropy, hash256, histogram, header_hash], dim=1)
    labels = torch.tensor(labels, dtype=torch.long)

    # 特征标准化
    scaler = StandardScaler()
    features_np = scaler.fit_transform(features.numpy())
    features = torch.tensor(features_np, dtype=torch.float32)

    # 特征选择（可选）
    # 使用随机森林评估特征重要性
    print("进行特征选择...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features_np[:10000], labels[:10000].numpy())
    importances = rf.feature_importances_
    important_indices = np.argsort(importances)[-800:]  # 选择前 800 个重要特征
    features = features[:, important_indices]

    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")

    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 创建训练集和验证集
    dataset = MalwareDataset(features, labels)
    val_size = int(VALID_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # =========================
    # 模型、损失函数、优化器
    # =========================
    model = MalwareModel(input_size=features.shape[1])
    model.apply(weights_init)
    model.to(DEVICE)

    # 使用类别权重
    model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # =========================
    # 早停机制
    # =========================
    class EarlyStopping:
        def __init__(self, patience=10, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.delta = delta
            self.best_score = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            torch.save(model.state_dict(), 'best_model.pt')
            if self.verbose:
                print('Validation loss decreased.  Saving model ...')

    early_stopping = EarlyStopping(patience=10, verbose=True)

    # =========================
    # 训练循环
    # =========================
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"第 {epoch + 1}/{NUM_EPOCHS} 轮")

        train_loss, train_acc, train_f1 = train_model(model, train_loader, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, DEVICE)

        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | 训练 F1 分数: {train_f1:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | 验证 F1 分数: {val_f1:.4f}")

        # 调度器步进
        scheduler.step(val_loss)

        # 早停
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("提前停止训练")
            break

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_malware_model.pth')
            print("已保存最佳模型。")

    print("训练完成。")

    # =========================
    # 加载最佳模型进行评估
    # =========================
    print("加载最佳模型进行评估...")
    model.load_state_dict(torch.load('best_malware_model.pth'))
    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, DEVICE)
    print(f"最佳验证损失: {val_loss:.4f} | 最佳验证准确率: {val_acc:.4f} | 最佳验证 F1 分数: {val_f1:.4f}")

    # =========================
    # 加载测试集并进行评估
    # =========================
    print("加载和处理测试数据中...")
    test_reader = ReadData([TEST_DATA_PATH], num_samples=None)  # 只读取测试文件
    test_texts, test_labels = test_reader.split_data()

    # 特征提取
    test_entropy = FirstFeature(test_texts).byteentropy()
    test_hash256 = SecFeature(test_texts).hash_to_256()
    test_histogram = ThiFeature(test_texts).get_histogram()
    test_header_hash = FourFeature(test_texts).hash_to_256()

    # 转换为张量
    test_entropy = torch.tensor(test_entropy, dtype=torch.float32)
    test_hash256 = torch.tensor(test_hash256, dtype=torch.float32)
    test_histogram = torch.tensor(test_histogram, dtype=torch.float32)
    test_header_hash = torch.tensor(test_header_hash, dtype=torch.float32)

    # 拼接特征
    test_features = torch.cat([test_entropy, test_hash256, test_histogram, test_header_hash], dim=1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 特征标准化（使用训练集的scaler）
    test_features_np = scaler.transform(test_features.numpy())
    test_features = torch.tensor(test_features_np, dtype=torch.float32)

    # 特征选择（使用训练集的重要特征索引）
    test_features = test_features[:, important_indices]

    # 创建测试集
    test_dataset = MalwareDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # =========================
    # 测试集评估
    # =========================
    print("在测试集上进行评估...")
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, DEVICE)
    print(f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f} | 测试 F1 分数: {test_f1:.4f}")

    # =========================
    # 混淆矩阵
    # =========================
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels_batch in tqdm(test_loader, desc="生成测试集混淆矩阵", leave=False):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("测试集混淆矩阵:")
    print(cm)


if __name__ == "__main__":
    main()
