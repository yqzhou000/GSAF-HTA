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


# ==================== 复用原始代码的组件 ====================

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


# ==================== 消融实验模型变体 ====================

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


class DrugEncoder_NoMultiScale(nn.Module):
    """药物编码器 - 移除多尺度CNN（仅使用单一kernel）"""
    def __init__(self, vocab_size, graph_feat_dim, embedding_dim=128, num_filters=64, graph_hidden=64):
        super(DrugEncoder_NoMultiScale, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 仅使用单一kernel size
        self.conv1 = nn.Conv1d(embedding_dim, num_filters * 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters * 3)
        
        self.attention = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(0.3)
        self.graph_encoder = GraphEncoder(graph_feat_dim, hidden_dim=graph_hidden)
        self.output_dim = num_filters * 3 + self.graph_encoder.output_dim
        self.output_dim = num_filters * 3
        self.graph_encoder = GraphEncoder(graph_feat_dim, hidden_dim=graph_hidden)
        self.output_dim = num_filters * 3 + self.graph_encoder.output_dim
        self.graph_encoder = GraphEncoder(graph_feat_dim, hidden_dim=graph_hidden)
        self.output_dim = num_filters * 3 + self.graph_encoder.output_dim
    
    def forward(self, x, graph_x, graph_adj, graph_mask):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv_out = F.relu(self.bn1(self.conv1(embedded))).transpose(1, 2)
        
        attention_weights = torch.softmax(self.attention(conv_out), dim=1)
        attended = (conv_out * attention_weights).sum(dim=1)
        
        seq_features = self.dropout(attended)
        graph_features = self.graph_encoder(graph_x, graph_adj, graph_mask)
        return torch.cat([seq_features, graph_features], dim=1)


class DrugEncoder_NoAttention(nn.Module):
    """药物编码器 - 移除注意力机制"""
    def __init__(self, vocab_size, graph_feat_dim, embedding_dim=128, num_filters=64, graph_hidden=64):
        super(DrugEncoder_NoAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.dropout = nn.Dropout(0.3)
        self.output_dim = lstm_hidden * 2
    
    def forward(self, x, graph_x, graph_adj, graph_mask):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        
        # 使用max pooling替代注意力
        pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
        seq_features = self.dropout(pooled)
        graph_features = self.graph_encoder(graph_x, graph_adj, graph_mask)
        return torch.cat([seq_features, graph_features], dim=1)


class ProteinEncoder_NoLSTM(nn.Module):
    """蛋白质编码器 - 移除LSTM"""
    def __init__(self, vocab_size, graph_feat_dim, embedding_dim=128, num_filters=64, graph_hidden=64):
        super(ProteinEncoder_NoLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.attention = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(0.3)
    
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


class ProteinEncoder_NoAttention(nn.Module):
    """蛋白质编码器 - 移除注意力机制"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64, lstm_hidden=128):
        super(ProteinEncoder_NoAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(
            num_filters * 3,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        lstm_out, (h_n, _) = self.lstm(conv_out)
        
        # 使用最后的隐藏状态
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        return self.dropout(final_hidden)


# ==================== 原始完整组件（用于baseline） ====================

class DrugEncoder(nn.Module):
    """药物编码器 - 完整版"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64):
        super(DrugEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.attention = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        attention_weights = torch.softmax(self.attention(conv_out), dim=1)
        attended = (conv_out * attention_weights).sum(dim=1)
        
        return self.dropout(attended)


class ProteinEncoder(nn.Module):
    """蛋白质编码器 - 完整版"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64, lstm_hidden=128):
        super(ProteinEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(
            num_filters * 3,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
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


# ==================== 消融实验预测器变体 ====================

class DTAPredictor_Baseline(nn.Module):
    """完整模型 - Baseline"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Baseline, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
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


class DTAPredictor_Ablation1(nn.Module):
    """消融1: 移除药物多尺度CNN"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation1, self).__init__()
        
        self.drug_encoder = DrugEncoder_NoMultiScale(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
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
        
        return self.fc_out(x)


class DTAPredictor_Ablation2(nn.Module):
    """消融2: 移除药物注意力"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation2, self).__init__()
        
        self.drug_encoder = DrugEncoder_NoAttention(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
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
        
        return self.fc_out(x)


class DTAPredictor_Ablation3(nn.Module):
    """消融3: 移除蛋白质LSTM"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation3, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder_NoLSTM(protein_vocab_size)
        
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
        
        return self.fc_out(x)


class DTAPredictor_Ablation4(nn.Module):
    """消融4: 移除蛋白质注意力"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation4, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder_NoAttention(protein_vocab_size)
        
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
        
        return self.fc_out(x)


class DTAPredictor_Ablation5(nn.Module):
    """消融5: 简化预测头（2层）"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation5, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
        combined_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim

        self.cross_attention = nn.MultiheadAttention(
            combined_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(combined_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(128, 1)
    
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
        
        return self.fc_out(x)


class DTAPredictor_Ablation6(nn.Module):
    """消融6: 移除Batch Normalization"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation6, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
        combined_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim

        self.cross_attention = nn.MultiheadAttention(
            combined_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc_out = nn.Linear(128, 1)
    
    def forward(self, drug, protein, graph_x, graph_adj, graph_mask):
        drug_features = self.drug_encoder(drug, graph_x, graph_adj, graph_mask)
        protein_features = self.protein_encoder(protein)
        
        combined = torch.cat([drug_features, protein_features], dim=1)
        combined_seq = combined.unsqueeze(1)
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)
        enhanced = combined + attended
        
        x = F.relu(self.fc1(enhanced))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        return self.fc_out(x)


class DTAPredictor_Ablation7(nn.Module):
    """消融7: 移除Dropout"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor_Ablation7, self).__init__()
        
        self.drug_encoder = DrugEncoder(drug_vocab_size, graph_feat_dim=graph_feat_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        
        combined_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim

        self.cross_attention = nn.MultiheadAttention(
            combined_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc_out = nn.Linear(128, 1)
    
    def forward(self, drug, protein, graph_x, graph_adj, graph_mask):
        drug_features = self.drug_encoder(drug, graph_x, graph_adj, graph_mask)
        protein_features = self.protein_encoder(protein)
        
        combined = torch.cat([drug_features, protein_features], dim=1)
        combined_seq = combined.unsqueeze(1)
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)
        enhanced = combined + attended
        
        x = F.relu(self.bn1(self.fc1(enhanced)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        
        return self.fc_out(x)


# ==================== 训练函数 ====================

def train_ablation_model(model, train_loader, val_loader, 
                        model_name, epochs=200, lr=0.001, device='cuda'):
    """训练单个消融模型"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    
    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_ci = 0
    patience_counter = 0
    patience = 30
    
    print(f"\n{'='*70}")
    print(f"训练模型: {model_name}")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # 训练
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
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        scheduler.step()
        
        # Early stopping
        if val_metrics['CI'] > best_ci:
            best_ci = val_metrics['CI']
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 
                      f'/root/1/result/ablation_{model_name}_1.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val CI: {val_metrics['CI']:.4f} | "
                  f"Rm²: {val_metrics['Rm2']:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"最佳 CI: {best_ci:.4f}\n")
    return best_ci


# ==================== 主函数 ====================

def main():
    """运行完整消融实验"""
    
    print("\n" + "="*70)
    print(" "*20 + "DTA模型消融实验")
    print("="*70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据
    print("📊 加载数据...")
    df = pd.read_excel('/root/1/data_pIC50.xlsx')
    df = df[(df['pIC50'] >= 0) & (df['pIC50'] <= 15)]
    print(f"数据集大小: {len(df)} 条\n")
    
    # 编码器
    smiles_encoder = SMILESEncoder(max_length=100)
    protein_encoder = ProteinSeqEncoder(max_length=1000)
    graph_builder = MolecularGraphBuilder(max_atoms=100)
    
    # 划分数据
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 创建数据加载器
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义所有消融实验
    ablation_experiments = [
        ('Baseline', DTAPredictor_Baseline),
        ('NoMultiScale_Drug', DTAPredictor_Ablation1),
        ('NoAttention_Drug', DTAPredictor_Ablation2),
        ('NoLSTM_Protein', DTAPredictor_Ablation3),
        ('NoAttention_Protein', DTAPredictor_Ablation4),
        ('Simplified_Predictor', DTAPredictor_Ablation5),
        ('NoBatchNorm', DTAPredictor_Ablation6),
        ('NoDropout', DTAPredictor_Ablation7),
    ]
    
    results = []
    
    # 运行所有实验
    for exp_name, ModelClass in ablation_experiments:
        print(f"\n🔬 开始实验: {exp_name}")
        print("-" * 70)
        
        model = ModelClass(
            drug_vocab_size=smiles_encoder.vocab_size,
            protein_vocab_size=protein_encoder.vocab_size,
            graph_feat_dim=graph_builder.num_atom_types
        ).to(device)
        
        # 训练
        best_ci = train_ablation_model(
            model, train_loader, val_loader,
            exp_name, epochs=200, lr=0.001, device=device
        )
        
        # 加载最佳模型并测试
        model.load_state_dict(
            torch.load(f'/root/1/result/ablation_{exp_name}_1.pt')
        )
        test_metrics = evaluate_model(model, test_loader, device)
        
        results.append({
            'Experiment': exp_name,
            'Test_MSE': test_metrics['MSE'],
            'Test_CI': test_metrics['CI'],
            'Test_Rm2': test_metrics['Rm2'],
            'Best_Val_CI': best_ci
        })
        
        print(f"✅ {exp_name} 完成:")
        print(f"   测试 MSE: {test_metrics['MSE']:.4f}")
        print(f"   测试 CI:  {test_metrics['CI']:.4f}")
        print(f"   测试 Rm²: {test_metrics['Rm2']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('/root/1/result/ablation_results_1.csv', index=False)
    
    # 显示汇总结果
    print("\n" + "="*70)
    print(" "*20 + "消融实验结果汇总")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # 计算性能下降
    baseline_ci = results_df[results_df['Experiment'] == 'Baseline']['Test_CI'].values[0]
    baseline_rm2 = results_df[results_df['Experiment'] == 'Baseline']['Test_Rm2'].values[0]
    
    print("\n相比Baseline的性能变化:")
    print(f"{'实验':<25} {'CI下降':<12} {'Rm²下降':<12}")
    print("-" * 50)
    
    for _, row in results_df.iterrows():
        if row['Experiment'] != 'Baseline':
            ci_drop = baseline_ci - row['Test_CI']
            rm2_drop = baseline_rm2 - row['Test_Rm2']
            print(f"{row['Experiment']:<25} {ci_drop:>+.4f}      {rm2_drop:>+.4f}")
    
    print("\n✅ 消融实验完成！结果已保存至: /root/1/result/ablation_results_1.csv\n")


if __name__ == "__main__":
    main()