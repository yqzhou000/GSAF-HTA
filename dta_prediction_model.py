import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ======================== 数据处理模块 ========================

class SMILESEncoder:
    """SMILES分子编码器"""
    def __init__(self, max_length=100):
        self.max_length = max_length
        # SMILES字符集
        self.char_set = set("CNOSPFIBrClcnos@=#()[]+-./\\1234567890")
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.char_set))}
        self.char_to_idx['<PAD>'] = 0
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, smiles):
        """将SMILES编码为数字序列"""
        encoded = []
        for char in smiles[:self.max_length]:
            encoded.append(self.char_to_idx.get(char, 0))
        # Padding
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class ProteinEncoder:
    """蛋白质序列编码器"""
    def __init__(self, max_length=1000):
        self.max_length = max_length
        # 标准氨基酸
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['<PAD>'] = 0
        self.aa_to_idx['X'] = len(self.aa_to_idx)  # 未知氨基酸
        self.vocab_size = len(self.aa_to_idx)
    
    def encode(self, sequence):
        """将蛋白质序列编码为数字序列"""
        encoded = []
        for aa in sequence[:self.max_length]:
            encoded.append(self.aa_to_idx.get(aa, self.aa_to_idx['X']))
        # Padding
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class DTADataset(Dataset):
    """DTA数据集"""
    def __init__(self, smiles_list, sequence_list, affinity_list, 
                 smiles_encoder, protein_encoder):
        self.smiles_list = smiles_list
        self.sequence_list = sequence_list
        self.affinity_list = affinity_list
        self.smiles_encoder = smiles_encoder
        self.protein_encoder = protein_encoder
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_encoder.encode(self.smiles_list[idx])
        protein = self.protein_encoder.encode(self.sequence_list[idx])
        affinity = self.affinity_list[idx]
        
        return {
            'smiles': torch.LongTensor(smiles),
            'protein': torch.LongTensor(protein),
            'affinity': torch.FloatTensor([affinity])
        }


# ======================== 模型架构模块 ========================

class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        # weighted sum
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights


class DrugEncoder(nn.Module):
    """药物编码器: Embedding + 1D CNN + Attention"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=32, kernel_sizes=[3, 5, 7]):
        super(DrugEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for k in kernel_sizes
        ])
        
        self.output_dim = num_filters * len(kernel_sizes)
        
        # Attention
        self.attention = AttentionLayer(self.output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(embedded))
        
        # 拼接
        conv_output = torch.cat(conv_outputs, dim=1)  # (batch, num_filters*3, seq_len)
        conv_output = conv_output.transpose(1, 2)  # (batch, seq_len, num_filters*3)
        
        # Attention pooling
        context, _ = self.attention(conv_output)
        
        return context


class ProteinEncoderModel(nn.Module):
    """蛋白质编码器: Embedding + 1D CNN + Bi-LSTM + Attention"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=32, 
                 lstm_hidden=64, kernel_sizes=[3, 5, 7]):
        super(ProteinEncoderModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for k in kernel_sizes
        ])
        
        conv_output_dim = num_filters * len(kernel_sizes)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(conv_output_dim, lstm_hidden, 
                           num_layers=2, bidirectional=True, 
                           batch_first=True, dropout=0.2)
        
        self.output_dim = lstm_hidden * 2  # bidirectional
        
        # Attention
        self.attention = AttentionLayer(self.output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(embedded))
        
        # 拼接
        conv_output = torch.cat(conv_outputs, dim=1)  # (batch, num_filters*3, seq_len)
        conv_output = conv_output.transpose(1, 2)  # (batch, seq_len, num_filters*3)
        
        # Bi-LSTM
        lstm_output, _ = self.lstm(conv_output)  # (batch, seq_len, lstm_hidden*2)
        
        # Attention pooling
        context, _ = self.attention(lstm_output)
        
        return context


class DTAPredictor(nn.Module):
    """完整的DTA预测模型"""
    def __init__(self, drug_vocab_size, protein_vocab_size):
        super(DTAPredictor, self).__init__()
        
        # 编码器
        self.drug_encoder = DrugEncoder(drug_vocab_size, embedding_dim=128, num_filters=32)
        self.protein_encoder = ProteinEncoderModel(protein_vocab_size, embedding_dim=128, 
                                             num_filters=32, lstm_hidden=64)
        
        # 融合层维度
        fusion_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim
        
        # 交互层 - 多头注意力
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=4, 
                                                     dropout=0.2, batch_first=True)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, drug, protein):
        # 编码
        drug_features = self.drug_encoder(drug)  # (batch, drug_dim)
        protein_features = self.protein_encoder(protein)  # (batch, protein_dim)
        
        # 拼接特征
        combined = torch.cat([drug_features, protein_features], dim=1)  # (batch, fusion_dim)
        
        # 添加序列维度用于注意力机制
        combined_seq = combined.unsqueeze(1)  # (batch, 1, fusion_dim)
        
        # 自注意力增强
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)  # (batch, fusion_dim)
        
        # 残差连接
        enhanced = combined + attended
        
        # 预测
        output = self.predictor(enhanced)
        
        return output


# ======================== 评估指标模块 ========================

def concordance_index(y_true, y_pred):
    """计算一致性指数 (Concordance Index, CI)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    concordant = 0
    total = 0
    
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
    
    if total == 0:
        return 0.5
    return concordant / total


def rm2_score(y_true, y_pred):
    """计算Rm² (modified r²)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Pearson correlation
    r, _ = pearsonr(y_true, y_pred)
    r2 = r ** 2
    
    # 计算均值
    y_mean = np.mean(y_true)
    
    # 计算Rm²
    k = np.sum((y_true - y_mean) * (y_pred - y_mean)) / np.sum((y_pred - y_mean) ** 2)
    
    # 修正的r²
    rm2 = r2 * (1 - np.sqrt(np.abs(r2 - k**2 * r2)))
    
    return rm2


def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            drug = batch['smiles'].to(device)
            protein = batch['protein'].to(device)
            affinity = batch['affinity'].to(device)
            
            predictions = model(drug, protein)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(affinity.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # 计算指标
    mse = mean_squared_error(all_labels, all_preds)
    ci = concordance_index(all_labels, all_preds)
    rm2 = rm2_score(all_labels, all_preds)
    
    return {
        'MSE': mse,
        'CI': ci,
        'Rm2': rm2
    }


# ======================== 训练模块 ========================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        drug = batch['smiles'].to(device)
        protein = batch['protein'].to(device)
        affinity = batch['affinity'].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(drug, protein)
        loss = criterion(predictions, affinity)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_model(model, train_loader, val_loader, epochs=100, 
                lr=0.0001, device='cuda', patience=15):
    """完整训练流程"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_ci = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_mse': [],
        'val_ci': [],
        'val_rm2': []
    }
    
    print("开始训练...\n")
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_mse'].append(val_metrics['MSE'])
        history['val_ci'].append(val_metrics['CI'])
        history['val_rm2'].append(val_metrics['Rm2'])
        
        # 学习率调度
        scheduler.step(val_metrics['MSE'])
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MSE: {val_metrics['MSE']:.4f}, CI: {val_metrics['CI']:.4f}, Rm²: {val_metrics['Rm2']:.4f}")
        
        # 早停检查
        if val_metrics['CI'] > best_ci:
            best_ci = val_metrics['CI']
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), '/root/1/result/best_dta_model.pt')
            print(f"  ✓ 新的最佳模型! CI: {best_ci:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停! 最佳epoch: {best_epoch+1}, 最佳CI: {best_ci:.4f}")
                break
        
        print()
    
    return history


# ======================== 主函数 ========================

def main():
    """主训练流程"""
    
    print("="*60)
    print("DTA预测模型训练")
    print("目标指标: CI >= 0.905, MSE <= 0.118, Rm² >= 0.812")
    print("="*60)
    print()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据
    print("加载数据...")
    df = pd.read_excel('/mnt/user-data/uploads/data_pIC50_1.xlsx')
    print(f"数据集大小: {len(df)}")
    print(f"pIC50 范围: [{df['pIC50'].min():.2f}, {df['pIC50'].max():.2f}]\n")
    
    # 数据清洗 - 移除异常值
    df = df[(df['pIC50'] >= 0) & (df['pIC50'] <= 15)]
    print(f"清洗后数据集大小: {len(df)}\n")
    
    # 初始化编码器
    smiles_encoder = SMILESEncoder(max_length=100)
    protein_encoder_obj = ProteinEncoder(max_length=1000)
    
    # 划分数据集
    print("划分数据集...")
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"测试集: {len(test_data)}\n")
    
    # 创建数据加载器
    train_dataset = DTADataset(
        train_data['Smiles'].tolist(),
        train_data['Sequence'].tolist(),
        train_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder_obj
    )
    
    val_dataset = DTADataset(
        val_data['Smiles'].tolist(),
        val_data['Sequence'].tolist(),
        val_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder_obj
    )
    
    test_dataset = DTADataset(
        test_data['Smiles'].tolist(),
        test_data['Sequence'].tolist(),
        test_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder_obj
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 创建模型
    print("创建模型...")
    model = DTAPredictor(
        drug_vocab_size=smiles_encoder.vocab_size,
        protein_vocab_size=protein_encoder_obj.vocab_size
    ).to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")
    
    # 训练模型
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=100,
        lr=0.0001,
        device=device,
        patience=15
    )
    
    # 加载最佳模型并在测试集上评估
    print("\n" + "="*60)
    print("在测试集上评估最佳模型...")
    model.load_state_dict(torch.load('/home/claude/best_dta_model.pt'))
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n最终测试集结果:")
    print(f"  MSE: {test_metrics['MSE']:.4f} (目标: <= 0.118)")
    print(f"  CI:  {test_metrics['CI']:.4f} (目标: >= 0.905)")
    print(f"  Rm²: {test_metrics['Rm2']:.4f} (目标: >= 0.812)")
    
    # 检查是否达到目标
    print("\n目标达成情况:")
    print(f"  MSE: {'✓' if test_metrics['MSE'] <= 0.118 else '✗'}")
    print(f"  CI:  {'✓' if test_metrics['CI'] >= 0.905 else '✗'}")
    print(f"  Rm²: {'✓' if test_metrics['Rm2'] >= 0.812 else '✗'}")
    print("="*60)
    
    return model, history, test_metrics


if __name__ == "__main__":
    model, history, test_metrics = main()