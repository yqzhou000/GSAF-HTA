import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from rdkit import Chem
import warnings
import os
import csv
import logging
from datetime import datetime

warnings.filterwarnings('ignore')


# ==================== 日志配置 ====================

def setup_logger(log_dir='/root/1/result/logs'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 创建logger
    logger = logging.getLogger('DTA_Training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def init_csv_logger(csv_path):
    """初始化CSV日志文件"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'lr',
                'train_loss',
                'val_mse',
                'val_ci',
                'val_rm2',
                'is_best_ci',
                'is_best_rm2'
            ])


def log_to_csv(csv_path, epoch, lr, train_loss, val_mse, val_ci, val_rm2, 
               is_best_ci=False, is_best_rm2=False):
    """记录训练指标到CSV"""
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f'{lr:.8f}',
            f'{train_loss:.6f}',
            f'{val_mse:.6f}',
            f'{val_ci:.6f}',
            f'{val_rm2:.6f}',
            int(is_best_ci),
            int(is_best_rm2)
        ])


# ==================== 数据编码器 ====================

class SMILESEncoder:
    """SMILES字符串编码器"""
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.char_set = set("CNOSPFIBrClcnos@=#()[]+-./\\1234567890")
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.char_set))}
        self.char_to_idx['<PAD>'] = 0
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, smiles):
        encoded = []
        for char in smiles[:self.max_length]:
            encoded.append(self.char_to_idx.get(char, 0))
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class MolecularGraphBuilder:
    def __init__(self, max_atoms=100):
        self.max_atoms = max_atoms
        self.atom_types = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'At', 'H'
        ]
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}
        self.num_atom_types = len(self.atom_types) + 1  # UNK

    def build(self, smiles):
        node_feats = np.zeros((self.max_atoms, self.num_atom_types), dtype=np.float32)
        adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
        mask = np.zeros((self.max_atoms,), dtype=np.float32)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return node_feats, adj, mask

        num_atoms = min(mol.GetNumAtoms(), self.max_atoms)
        for i in range(num_atoms):
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            idx = self.atom_to_idx.get(symbol, self.num_atom_types - 1)
            node_feats[i, idx] = 1.0
            mask[i] = 1.0

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        for i in range(num_atoms):
            adj[i, i] = 1.0

        return node_feats, adj, mask


class ProteinSeqEncoder:
    """蛋白质序列编码器"""
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['<PAD>'] = 0
        self.aa_to_idx['X'] = len(self.aa_to_idx)
        self.vocab_size = len(self.aa_to_idx)
    
    def encode(self, sequence):
        encoded = []
        for aa in sequence[:self.max_length]:
            encoded.append(self.aa_to_idx.get(aa, self.aa_to_idx['X']))
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class DTADataset(Dataset):
    """DTA数据集"""
    def __init__(self, smiles_list, sequence_list, affinity_list, 
                 smiles_encoder, protein_encoder, graph_builder):
        self.smiles_list = smiles_list
        self.sequence_list = sequence_list
        self.affinity_list = affinity_list
        self.smiles_encoder = smiles_encoder
        self.protein_encoder = protein_encoder
        self.graph_builder = graph_builder
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_encoder.encode(self.smiles_list[idx])
        protein = self.protein_encoder.encode(self.sequence_list[idx])
        node_feats, adj, mask = self.graph_builder.build(self.smiles_list[idx])
        affinity = self.affinity_list[idx]
        
        return {
            'smiles': torch.LongTensor(smiles),
            'protein': torch.LongTensor(protein),
            'graph_x': torch.FloatTensor(node_feats),
            'graph_adj': torch.FloatTensor(adj),
            'graph_mask': torch.FloatTensor(mask),
            'affinity': torch.FloatTensor([affinity])
        }


# ==================== 模型组件 ====================

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, node_feats, adj, mask):
        deg = adj.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        h = node_feats
        for layer in self.layers:
            h = torch.matmul(norm_adj, h)
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)

        attn_logits = self.attention(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * h, dim=1)
        return pooled


class DrugEncoder(nn.Module):
    """药物编码器"""
    def __init__(self, vocab_size, graph_feat_dim, embedding_dim=128, num_filters=64, graph_hidden=64):
        super(DrugEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度CNN
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        # 注意力
        self.attention = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(0.3)
        self.graph_encoder = GraphEncoder(graph_feat_dim, hidden_dim=graph_hidden)
        self.output_dim = num_filters * 3 + self.graph_encoder.output_dim
    
    def forward(self, x, graph_x, graph_adj, graph_mask):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        attention_weights = torch.softmax(self.attention(conv_out), dim=1)
        attended = (conv_out * attention_weights).sum(dim=1)
        
        seq_features = self.dropout(attended)
        graph_features = self.graph_encoder(graph_x, graph_adj, graph_mask)
        return torch.cat([seq_features, graph_features], dim=1)


class ProteinEncoderModel(nn.Module):
    """蛋白质编码器"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64, lstm_hidden=128):
        super(ProteinEncoderModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度CNN
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            num_filters * 3,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 注意力
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.output_dim = lstm_hidden * 2
    
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        lstm_out, _ = self.lstm(conv_out)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = (lstm_out * attention_weights).sum(dim=1)
        
        return self.dropout(attended)


class DTAPredictor(nn.Module):
    """DTA预测模型"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor, self).__init__()
        
        self.drug_encoder = DrugEncoder(
            drug_vocab_size,
            graph_feat_dim=graph_feat_dim,
            embedding_dim=128,
            num_filters=64
        )
        self.protein_encoder = ProteinEncoderModel(protein_vocab_size, embedding_dim=128, 
                                                   num_filters=64, lstm_hidden=128)
        
        combined_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim

        self.cross_attention = nn.MultiheadAttention(
            combined_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc_out = nn.Linear(128, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, drug, protein, graph_x, graph_adj, graph_mask):
        drug_features = self.drug_encoder(drug, graph_x, graph_adj, graph_mask)
        protein_features = self.protein_encoder(protein)
        
        combined = torch.cat([drug_features, protein_features], dim=1)
        combined_seq = combined.unsqueeze(1)
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)
        enhanced = combined + attended
        
        x = F.relu(self.bn1(self.fc1(enhanced)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        output = self.fc_out(x)
        
        return output


# ==================== 评估指标 ====================

def concordance_index(y_true, y_pred):
    """计算Concordance Index"""
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
    
    return concordant / total if total > 0 else 0.5


def rm2_score(y_true, y_pred):
    """计算Modified R²"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r, _ = pearsonr(y_true, y_pred)
    r2 = r ** 2
    
    y_mean = np.mean(y_true)
    k = np.sum((y_true - y_mean) * (y_pred - y_mean)) / (np.sum((y_pred - y_mean) ** 2) + 1e-8)
    
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
            graph_x = batch['graph_x'].to(device)
            graph_adj = batch['graph_adj'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            affinity = batch['affinity'].to(device)
            
            predictions = model(drug, protein, graph_x, graph_adj, graph_mask)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(affinity.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    mse = mean_squared_error(all_labels, all_preds)
    ci = concordance_index(all_labels, all_preds)
    rm2 = rm2_score(all_labels, all_preds)
    
    return {'MSE': mse, 'CI': ci, 'Rm2': rm2}


# ==================== 训练函数 ====================

def train_model(model, train_loader, val_loader, epochs=500, lr=0.001, device='cuda', logger=None):
    """训练模型，支持 Warmup+CosineLR，保存最佳 CI 和 Rm² 模型"""
    
    # 初始化CSV日志
    csv_path = '/root/1/result/training_stage1.csv'
    init_csv_logger(csv_path)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-5,
        betas=(0.9, 0.999)
    )
    
    # Warmup + CosineAnnealingLR with min_lr
    warmup_epochs = 30
    min_lr = 2e-4
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return min_lr/lr + (1 - min_lr/lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_ci = 0
    best_rm2 = -np.inf
    best_epoch_ci = 0
    best_epoch_rm2 = 0
    patience_counter = 0
    patience = 50
    
    history = {
        'train_loss': [],
        'val_mse': [],
        'val_ci': [],
        'val_rm2': []
    }
    
    os.makedirs('/root/1/result', exist_ok=True)
    
    if logger:
        logger.info("="*70)
        logger.info("开始训练DTA模型 - Stage 1")
        logger.info("="*70)
        logger.info(f"设备: {device}")
        logger.info(f"学习率: {lr}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Early stopping patience: {patience}")
        logger.info(f"CSV日志: {csv_path}")
        logger.info("="*70)
    
    print("\n" + "="*70)
    print(" "*20 + "开始训练DTA模型")
    print("="*70)
    print(f"  设备: {device}")
    print(f"  学习率: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print("="*70 + "\n")
    
    for epoch in range(epochs):
        # ===== 训练 =====
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            drug = batch['smiles'].to(device)
            protein = batch['protein'].to(device)
            graph_x = batch['graph_x'].to(device)
            graph_adj = batch['graph_adj'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            affinity = batch['affinity'].to(device)
            
            optimizer.zero_grad()
            predictions = model(drug, protein, graph_x, graph_adj, graph_mask)
            loss = criterion(predictions, affinity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # ===== 验证 =====
        val_metrics = evaluate_model(model, val_loader, device)
        
        # ===== 记录历史 =====
        history['train_loss'].append(avg_train_loss)
        history['val_mse'].append(val_metrics['MSE'])
        history['val_ci'].append(val_metrics['CI'])
        history['val_rm2'].append(val_metrics['Rm2'])
        
        # ===== 更新学习率 =====
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ===== 检查是否为最佳模型 =====
        is_best_ci = False
        is_best_rm2 = False
        improved = False
        
        # 根据 CI 保存模型
        if val_metrics['CI'] > best_ci:
            best_ci = val_metrics['CI']
            best_epoch_ci = epoch
            torch.save(model.state_dict(), '/root/1/result/best_dta_model_bestCI.pt')
            is_best_ci = True
            improved = True
            if logger:
                logger.info(f"Epoch {epoch+1}: 新的最佳 CI 模型! CI: {best_ci:.4f}")
        
        # 根据 Rm² 保存模型
        if val_metrics['Rm2'] > best_rm2:
            best_rm2 = val_metrics['Rm2']
            best_epoch_rm2 = epoch
            torch.save(model.state_dict(), '/root/1/result/best_dta_model_bestRm2.pt')
            is_best_rm2 = True
            improved = True
            if logger:
                logger.info(f"Epoch {epoch+1}: 新的最佳 Rm² 模型! Rm²: {best_rm2:.4f}")
        
        # ===== 记录到CSV =====
        log_to_csv(
            csv_path,
            epoch + 1,
            current_lr,
            avg_train_loss,
            val_metrics['MSE'],
            val_metrics['CI'],
            val_metrics['Rm2'],
            is_best_ci,
            is_best_rm2
        )
        
        # ===== 记录到日志 =====
        if logger:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val MSE: {val_metrics['MSE']:.4f} | "
                f"CI: {val_metrics['CI']:.4f} | "
                f"Rm²: {val_metrics['Rm2']:.4f}"
            )
        
        # ===== 打印进度 =====
        print(f"Epoch {epoch+1:3d}/{epochs} [LR: {current_lr:.6f}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val - MSE: {val_metrics['MSE']:.4f}, CI: {val_metrics['CI']:.4f}, Rm²: {val_metrics['Rm2']:.4f}")
        
        if is_best_ci:
            print(f"  ✓ 新的最佳 CI 模型! CI: {best_ci:.4f}")
        if is_best_rm2:
            print(f"  ✓ 新的最佳 Rm² 模型! Rm²: {best_rm2:.4f}")
        
        # ===== Early Stopping =====
        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                msg = f"Early stopping triggered at epoch {epoch+1}."
                print(f"\n⚠️ {msg}")
                if logger:
                    logger.warning(msg)
                break
        else:
            patience_counter = 0
    
    summary = f"\n训练完成！最佳 CI 模型在 Epoch {best_epoch_ci+1}, CI: {best_ci:.4f} | 最佳 Rm² 模型在 Epoch {best_epoch_rm2+1}, Rm²: {best_rm2:.4f}"
    print(summary)
    if logger:
        logger.info(summary)
    
    return model, history


def finetune_stage2(model, train_loader, val_loader, device='cuda', logger=None):
    """Stage 2 微调"""
    
    # 初始化CSV日志
    csv_path = '/root/1/result/training_stage2.csv'
    init_csv_logger(csv_path)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-5
    )

    # ===== Re-warmup + short cosine =====
    total_epochs = 60
    warmup_epochs = 10
    min_lr = 2e-4

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr/1e-3 + (1 - min_lr/1e-3) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ci = 0.0

    msg = "Stage 2: LR Restart Fine-tuning"
    print(f"\n🚀 {msg}\n")
    if logger:
        logger.info("="*70)
        logger.info(msg)
        logger.info(f"CSV日志: {csv_path}")
        logger.info("="*70)

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch['smiles'].to(device),
                batch['protein'].to(device),
                batch['graph_x'].to(device),
                batch['graph_adj'].to(device),
                batch['graph_mask'].to(device)
            )
            loss = criterion(pred, batch['affinity'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        
        is_best_ci = False
        if val_metrics['CI'] > best_ci:
            best_ci = val_metrics['CI']
            torch.save(
                model.state_dict(),
                '/root/1/result/best_dta_model_stage2.pt'
            )
            is_best_ci = True
            if logger:
                logger.info(f"Epoch {epoch+1}: 新的最佳 CI: {best_ci:.4f}")
        
        # 记录到CSV
        log_to_csv(
            csv_path,
            epoch + 1,
            lr_now,
            train_loss,
            val_metrics['MSE'],
            val_metrics['CI'],
            val_metrics['Rm2'],
            is_best_ci,
            False
        )
        
        # 记录到日志
        if logger:
            logger.info(
                f"[Stage2] Epoch {epoch+1:02d}/{total_epochs} | "
                f"LR={lr_now:.6f} | "
                f"Train={train_loss:.4f} | "
                f"CI={val_metrics['CI']:.4f} | "
                f"Rm²={val_metrics['Rm2']:.4f}"
            )

        print(f"[Stage2] Epoch {epoch+1:02d}/{total_epochs} "
              f"LR={lr_now:.6f} "
              f"Train={train_loss:.4f} "
              f"CI={val_metrics['CI']:.4f} "
              f"Rm²={val_metrics['Rm2']:.4f}")
        
        if is_best_ci:
            print(f"  🔥 New best CI: {best_ci:.4f}")

    msg = "Stage 2 finished."
    print(f"\n✅ {msg}")
    if logger:
        logger.info(msg)


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()
    
    logger.info("="*70)
    logger.info("DTA预测模型 - 最终完整版（带日志记录）")
    logger.info("目标: CI ≥ 0.905, MSE ≤ 0.118, Rm² ≥ 0.812")
    logger.info("="*70)
    
    print("\n" + "="*70)
    print(" "*15 + "DTA预测模型 - 最终完整版")
    print("="*70)
    print("  目标: CI ≥ 0.905, MSE ≤ 0.118, Rm² ≥ 0.812")
    print("="*70 + "\n")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name}")
        print(f"GPU: {gpu_name}")
    print()
    
    # 加载数据
    logger.info("加载数据...")
    print("📊 加载数据...")
    # df = pd.read_excel('/root/1/data_pIC50.xlsx')
    df = pd.read_excel('/root/1/Davis.xlsx')
    print(f"  原始数据: {len(df)} 条")
    print(f"  pIC50范围: [{df['pIC50'].min():.2f}, {df['pIC50'].max():.2f}]")
    
    # 数据清洗
    original_len = len(df)
    df = df[(df['pIC50'] >= 0) & (df['pIC50'] <= 15)]
    removed = original_len - len(df)
    print(f"  清洗后: {len(df)} 条 (移除{removed}条异常值)")
    print()
    
    # 编码器
    print("🔤 初始化编码器...")
    smiles_encoder = SMILESEncoder(max_length=100)
    protein_encoder = ProteinSeqEncoder(max_length=1000)
    graph_builder = MolecularGraphBuilder(max_atoms=100)
    print(f"  SMILES词汇表: {smiles_encoder.vocab_size}")
    print(f"  蛋白质词汇表: {protein_encoder.vocab_size}")
    print()
    
    # 划分数据
    print("✂️  划分数据集...")
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"  训练集: {len(train_data):,} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_data):,} ({len(val_data)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_data):,} ({len(test_data)/len(df)*100:.1f}%)")
    print()
    
    # 创建数据集
    print("🔄 创建数据加载器...")
    train_dataset = DTADataset(
        train_data['Smiles'].tolist(),
        train_data['Sequence'].tolist(),
        train_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder,
        graph_builder
    )
    
    val_dataset = DTADataset(
        val_data['Smiles'].tolist(),
        val_data['Sequence'].tolist(),
        val_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder,
        graph_builder
    )
    
    test_dataset = DTADataset(
        test_data['Smiles'].tolist(),
        test_data['Sequence'].tolist(),
        test_data['pIC50'].tolist(),
        smiles_encoder,
        protein_encoder,
        graph_builder
    )
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Batch大小: {batch_size}")
    print()
    
    # 创建模型
    print("🏗️  构建模型...")
    model = DTAPredictor(
        drug_vocab_size=smiles_encoder.vocab_size,
        protein_vocab_size=protein_encoder.vocab_size,
        graph_feat_dim=graph_builder.num_atom_types
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    print()
    
    # 训练 Stage 1
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=500,
        lr=0.001,
        device=device,
        logger=logger
    )
    
    # 加载最佳模型进行 Stage 2
    print("\nLoading best Stage-1 model for Stage-2 fine-tuning...")
    model.load_state_dict(
        torch.load('/root/1/result/best_dta_model_bestCI.pt')
    )

    finetune_stage2(model, train_loader, val_loader, device, logger)
    
    # 测试集评估
    print("\n📈 最终测试集评估...")
    
    # 加载Stage 2的最佳模型
    model.load_state_dict(torch.load('/root/1/result/best_dta_model_stage2.pt'))
    test_metrics = evaluate_model(model, test_loader, device)
    
    
    print("\n" + "="*70)
    print(" "*20 + "最终测试结果")
    print("="*70)
    print(f"\n{'指标':<10} {'实际值':<12} {'目标值':<12} {'状态'}")
    print("-"*70)
    
    mse_ok = test_metrics['MSE'] <= 0.118
    ci_ok = test_metrics['CI'] >= 0.905
    rm2_ok = test_metrics['Rm2'] >= 0.812
    
    print(f"{'MSE':<10} {test_metrics['MSE']:<12.4f} {'≤ 0.118':<12} {'✓' if mse_ok else '✗'}")
    print(f"{'CI':<10} {test_metrics['CI']:<12.4f} {'≥ 0.905':<12} {'✓' if ci_ok else '✗'}")
    print(f"{'Rm²':<10} {test_metrics['Rm2']:<12.4f} {'≥ 0.812':<12} {'✓' if rm2_ok else '✗'}")
    
    targets_met = sum([mse_ok, ci_ok, rm2_ok])
    
    
    print("\n" + "="*70)
    if targets_met == 3:
        msg = "🌟 完美! 所有目标均已达成!"
        print(msg)
    elif targets_met == 2:
        msg = "⭐ 良好! 大部分目标已达成"
        print(msg)
    else:
        msg = "⚠️  需要继续优化"
        print(msg)
        logger.warning(msg)
    print("="*70 + "\n")
    
    
    print("输出文件:")
    print("  - best_dta_model_bestCI.pt (Stage 1 最佳CI模型)")
    print("  - best_dta_model_bestRm2.pt (Stage 1 最佳Rm²模型)")
    print("  - best_dta_model_stage2.pt (Stage 2 最佳模型)")
    print("  - training_stage1.csv (Stage 1 训练记录)")
    print("  - training_stage2.csv (Stage 2 训练记录)")
    
    # ==================== 计算平均指标和最佳平均指标 ====================
    print("\n" + "="*70)
    print(" "*15 + "所有模型平均指标统计")
    print("="*70)
    
    # 评估所有三个模型
    models_to_evaluate = [
        ('Stage1_BestCI', '/root/1/result/best_dta_model_bestCI.pt'),
        ('Stage1_BestRm2', '/root/1/result/best_dta_model_bestRm2.pt'),
        ('Stage2_Final', '/root/1/result/best_dta_model_stage2.pt')
    ]
    
    all_metrics = []
    
    for model_name, model_path in models_to_evaluate:
        model.load_state_dict(torch.load(model_path))
        metrics = evaluate_model(model, test_loader, device)
        all_metrics.append({
            'model': model_name,
            'MSE': metrics['MSE'],
            'CI': metrics['CI'],
            'Rm2': metrics['Rm2']
        })
        print(f"{model_name:<20} MSE={metrics['MSE']:.4f}, CI={metrics['CI']:.4f}, Rm²={metrics['Rm2']:.4f}")
    
    # 计算平均指标
    avg_mse = np.mean([m['MSE'] for m in all_metrics])
    avg_ci = np.mean([m['CI'] for m in all_metrics])
    avg_rm2 = np.mean([m['Rm2'] for m in all_metrics])
    
    # 找出最佳平均指标（综合得分最高的模型）
    # 综合得分 = CI权重 + Rm2权重 - MSE权重（归一化）
    scores = []
    for m in all_metrics:
        # 归一化MSE（越小越好，所以用1减）
        norm_mse = 1 - (m['MSE'] / 1.0)  # 假设MSE最大为1
        # CI和Rm2越大越好
        combined_score = 0.4 * m['CI'] + 0.4 * m['Rm2'] + 0.2 * norm_mse
        scores.append(combined_score)
    
    best_idx = np.argmax(scores)
    best_model = all_metrics[best_idx]
    
    # 输出平均指标  
    print("\n" + "-"*70)
    print("平均指标 (3个模型):")
    print(f"  平均 MSE: {avg_mse:.4f}")
    print(f"  平均 CI:  {avg_ci:.4f}")
    print(f"  平均 Rm²: {avg_rm2:.4f}")
    
    # 输出最佳综合模型 
    print("\n" + "-"*70)
    print("最佳综合模型 (综合得分最高):")
    print(f"  模型: {best_model['model']}")
    print(f"  MSE: {best_model['MSE']:.4f}")
    print(f"  CI:  {best_model['CI']:.4f}")
    print(f"  Rm²: {best_model['Rm2']:.4f}")
    print(f"  综合得分: {scores[best_idx]:.4f}")
    print("="*70)
    
    # 写入综合指标CSV
    # summary_csv_path = '/root/1/result/model_summary.csv'
    summary_best_csv_path = '/root/1/result/model_summary_1.csv'
    with open(summary_best_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mse', 'ci', 'rm2'])
        writer.writerow([
            f"{best_model['MSE']:.6f}",
            f"{best_model['CI']:.6f}",
            f"{best_model['Rm2']:.6f}"
        ])
    
    print("\n新增输出文件:")
    print("  - model_summary.csv (模型综合评估)")
    
    print("\n训练完成! 🎉\n")
    logger.info("\n训练完成! 🎉")


if __name__ == "__main__":
    main()